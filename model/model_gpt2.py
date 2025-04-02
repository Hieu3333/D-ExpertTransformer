from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from modules.visual_extractor import ResNet50,EfficientNet
import sys
from collections import Counter
from modules.RMSNorm import RMSNorm
from modules.GCA import GuidedContextAttention




def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

    
class MultiHeadedAttention(nn.Module):
    def __init__(self,args, mask=True):
        super(MultiHeadedAttention,self).__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.head_size = self.hidden_size // self.num_heads
        self.dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        assert self.hidden_size % self.num_heads == 0
        self.mask = mask
        

        self.q_proj = nn.Linear(self.hidden_size,self.hidden_size,bias=args.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size,bias=args.bias)
        self.v_proj = nn.Linear(self.hidden_size,self.hidden_size,bias=args.bias)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size,bias=args.bias)
        if self.mask:
            self.register_buffer('bias',torch.tril(torch.ones(args.max_gen,args.max_gen).view(1,1,args.max_gen,args.max_gen))) 

    def forward(self,query,key,value):
        B,T,_ = query.shape #T is number of keywords
        B,N,_ = key.shape

        q = self.q_proj(query) #(B,T,C)
        k = self.k_proj(key) 
        v = self.v_proj(value) 
        

        q = q.view(B,T,self.num_heads,self.head_size).transpose(1,2) #(B,nh,T,head_size)
        k = k.view(B,N,self.num_heads,self.head_size).transpose(1,2) #(B,nh,N,head_size)
        v = v.view(B,N,self.num_heads,self.head_size).transpose(1,2) #(B,nh,N,head_size)
        
        assert q.shape[-1] == k.shape[-1]
        att = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(q.shape[-1]) #(B,nh,T,N)
        # print('att:',att.shape)
        if self.mask:
            att = att.masked_fill(
                    self.bias[:,:,:att.shape[2],:att.shape[3]] == 0, 
                    torch.finfo(att.dtype).min  # Ensures proper handling in mixed precision
                )
        att = F.softmax(att,dim=-1)
        att = self.dropout(att)
        out = torch.matmul(att,v) #(B,nh,T,T) @ (B,nh,T,head_size) -> (B,nh,T,head_size)
        out = out.transpose(1,2).contiguous().view(B,T,self.hidden_size)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out

class DiffMultiHeadedAttention(nn.Module):
    def __init__(self,args,depth,mask=True):
        super(DiffMultiHeadedAttention,self).__init__()
        self.hidden_size = args.hidden_size
        self.diff_num_heads = args.diff_num_heads
        self.diff_head_size = self.hidden_size // self.diff_num_heads
        self.dropout = nn.Dropout(args.dropout)
        self.mask = mask

        assert self.hidden_size % self.diff_num_heads == 0

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.q_proj = nn.Linear(self.hidden_size,self.hidden_size,bias=args.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size,bias=args.bias)
        self.v_proj = nn.Linear(self.hidden_size,self.hidden_size,bias=args.bias)

        self.rmsnorm = RMSNorm(self.diff_head_size, eps=1e-5, elementwise_affine=True)


        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size,bias=args.bias)
        self.register_buffer('bias',torch.tril(torch.ones(args.max_gen,args.max_gen).view(1,1,args.max_gen,args.max_gen))) 

    def forward(self,query,key,value):
        B,T,_ = query.shape #T is number of keywords
        B,N,_ = value.shape

        q = self.q_proj(query) #(B,T,C)
        k = self.k_proj(key) 
        v = self.v_proj(value) 
        
        lambda1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_full = lambda1 - lambda2 + self.lambda_init

        q = q.reshape(B,T,2*self.diff_num_heads,self.diff_head_size//2).transpose(1,2) #(B,2*nh,T,diff_head_size/2)
        k = k.reshape(B,N,2*self.diff_num_heads,self.diff_head_size//2).transpose(1,2) #(B,2*nh,N,diff_head_size/2)
        v = v.reshape(B,N,self.diff_num_heads,self.diff_head_size).transpose(1,2) #(B,nh,N,diff_head_size)
        
        assert q.shape[-1] == k.shape[-1]
        att = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(q.shape[-1]) #(B,nh,T,N)
        # print('att:',att.shape)
        if self.mask:
            att = att.masked_fill(
                    self.bias[:,:,:att.shape[2],:att.shape[3]] == 0, 
                    torch.finfo(att.dtype).min  # Ensures proper handling in mixed precision
                )
        att = F.softmax(att,dim=-1)
        att = att.reshape(B,self.diff_num_heads,2,T,-1)
        att = att[:,:,0] - lambda_full * att[:,:,1]
        out = torch.matmul(att,v) #(B,nh,T,T) @ (B,nh,T,head_size) -> (B,nh,T,head_size)
        out = self.rmsnorm(out) * (1-self.lambda_init)  # (B, nh, T, head_size)
        # out = out * (1-self.lambda_init)
        out = out.transpose(1,2).contiguous().view(B,T,self.hidden_size)
        out = self.out_proj(out) 
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self,args):
        super(MLP,self).__init__()
        self.c_fc = nn.Linear(args.hidden_size,args.hidden_size//2,bias=args.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(args.hidden_size//2,args.hidden_size,bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self,x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class TransfusionEncoder(nn.Module):
    def __init__(self,args,depth):
        super(TransfusionEncoder,self).__init__()
        self.attn = DiffMultiHeadedAttention(args,depth,mask=False)
        self.depth = depth
        if depth == 0:
            self.vf_proj = nn.Linear(args.encoder_size, args.hidden_size)
        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.mlp = MLP(args)
        self.ln2 = nn.LayerNorm(args.hidden_size)
    
    def forward(self,visual_features,x):   #args.hidden_size = gpt2 embed dim
        if self.depth == 0:
            vf = self.vf_proj(visual_features)
        else:
            vf = visual_features
        vf = self.ln1(vf + self.attn(vf,x,x))
        vf = self.ln2(vf +self.mlp(vf))
        return vf

class TokenFuser(nn.Module):
    def __init__(self,args,depth):
        self.attn = DiffMultiHeadedAttention(args,depth=depth,mask=False)
        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.mlp = MLP(args)
        self.ln2 = nn.LayerNorm(args.hidden_size)
    
    def forward(self,encoder_feat,token_embed):
        x = self.ln1(x + self.attn(token_embed,encoder_feat,encoder_feat))
        x = self.ln2(x + self.mlp(x))
        return x
    
class VisualEncoder(nn.Module):
    def __init__(self,args):
        super(VisualEncoder,self).__init__()
        if args.ve_name == 'resnet':
            self.ve = ResNet50()
        else:
            self.ve = EfficientNet()

        # self.gca = GuidedContextAttention(args)

    def forward(self,images):
        vf = self.ve(images)
        B,C,_,_ = vf.size()
        vf = vf.view(B,C,-1)
        return vf.transpose(-2,-1)





class ExpertTransformer(nn.Module):
    def __init__(self, args, tokenizer, keywords):
        super(ExpertTransformer, self).__init__()
        
        # Initialize GPT-2 model and freeze its parameters
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        for param in self.gpt2.parameters():
            param.requires_grad = False  # Freeze GPT-2 parameters
        
        self.tokenizer = tokenizer
        
        self.args = args
        self.max_length = args.max_length
        self.max_gen = args.max_gen
        self.threshold = args.threshold
        self.num_layers = args.num_layers
        self.delta1 = args.delta1
        self.delta2 = args.delta2
        self.topk = args.topk
        self.temperature = args.temperature
        
        self.dropout = nn.Dropout(args.dropout)
        
        # Visual encoder and feature fusion
        self.visual_encoder = VisualEncoder(args)  # assuming you have your visual encoder
        self.fuser = nn.ModuleList([TransfusionEncoder(args, depth=depth) for depth in range(args.num_layers)])
        self.token_fuser = nn.ModuleList([TokenFuser(args, depth=depth) for depth in range(args.num_layers)])
        
        # Language modeling head (output vocabulary logits)
        self.lm_head = nn.Linear(self.gpt2.config.n_embd, self.gpt2.config.vocab_size, bias=False)
        
        self.keywords = keywords
        self.device = args.device
        self.beam_width = args.beam_width
        
        # Weight tying: use GPT-2's embeddings
        self.lm_head.weight = self.gpt2.transformer.wte.weight

        for param in self.gpt2.parameters():
            param.requires_grad = False  # Freeze GPT-2 parameters
        
        # Apply initialization to all layers
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, images, tokens, gt_keyword_tokens, targets=None):
   
        B, T = tokens.shape  # Batch size, Sequence length
        device = tokens.device

        # 1. Extract visual encoder features
        visual_features = self.visual_encoder(images)  # (B, N, hidden_size)

        # 2. Process keywords (if any)
        keyword_emb = self.gpt2.transformer.wte(gt_keyword_tokens)  # (B, K, hidden_size)
        
        # 3. Fuse visual and keyword features
        encoder_feature = self.fuser[0](visual_features, keyword_emb)  # (B, N, hidden_size)
        tok_emb = self.gpt2.transformer.wte(tokens)  # (B, T, hidden_size)

        x = self.token_fuser[0](encoder_feature,tok_emb) #(B,T,C)

        pos_ids = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        pos_emb = self.gpt2.transformer.wpe(pos_ids)  # (1, T, hidden_size)
        
        # 7. Add positional embeddings
        x = x + pos_emb  # (B, T, hidden_size)

        # 8. Pass through GPT-2’s transformer layers
        transformer_outputs = self.gpt2.transformer(inputs_embeds=x)
        hidden_states = transformer_outputs.last_hidden_state  # (B, T, hidden_size)

        # 9. Compute logits
        logits = self.lm_head(hidden_states)  # (B, T, vocab_size)

        # 10. Compute loss if targets are provided
        loss_ce = None
        if targets is not None:
            loss_ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # Flatten logits
                targets.view(-1),  # Flatten targets
                ignore_index=self.tokenizer.pad_token_id  # Ignore padding tokens
            )

        return logits, loss_ce
    

    @torch.no_grad()
    def generate_beam(self, images, gt_keywords):
        device = self.device
        batch_size = images.size(0)
        beam_width = self.beam_width

        # Get the BOS and EOS token IDs from the GPT-2 tokenizer
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # Set pad_token_id to eos_token_id
        pad_token_id = eos_id

        # Initialize sequences and log probabilities
        sequences = torch.full((batch_size, 1), bos_id, device=device, dtype=torch.long)
        log_probs = torch.zeros(batch_size, device=device)  # Log probability of sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Beam candidates (track beam_width sequences per batch)
        beam_sequences = [sequences.clone() for _ in range(beam_width)]
        beam_scores = [log_probs.clone() for _ in range(beam_width)]

        for t in range(1, self.max_gen):
            all_candidates = []

            for i in range(beam_width):  # Iterate over beams
                logits, _ = self(images, beam_sequences[i], gt_keywords)  # (B, seq_len, vocab_size)
                logits = logits[:, -1, :] / self.temperature  # Get logits for last token
                probs = F.log_softmax(logits, dim=-1)  # Convert to log probabilities

                # Select top-k candidates (beam search step)
                top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)  # (B, beam_width)
                
                # Compute new sequence candidates
                for j in range(beam_width):
                    new_seq = torch.cat([beam_sequences[i], top_indices[:, j].unsqueeze(1)], dim=-1)
                    new_score = beam_scores[i] + top_probs[:, j]  # Update log probability score
                    all_candidates.append((new_seq, new_score))

            # Select the top beam_width sequences across all candidates
            all_candidates.sort(key=lambda x: x[1].sum().item(), reverse=True)  # Convert tensor to scalar for sorting
            beam_sequences = [x[0] for x in all_candidates[:beam_width]]
            beam_scores = [x[1] for x in all_candidates[:beam_width]]

            # Check if all sequences have reached EOS
            for i in range(beam_width):
                finished |= beam_sequences[i][:, -1] == eos_id
            if finished.all():
                break

        # Decode sequences
        final_sequences = []
        for i in range(batch_size):
            best_seq = beam_sequences[0][i].tolist()  # Take the highest-scoring sequence for each batch
            if eos_id in best_seq:
                best_seq = best_seq[:best_seq.index(eos_id) + 1]  # Trim sequence at EOS token
            text = self.tokenizer.decode(best_seq, skip_special_tokens=True)  # Decode using GPT-2 tokenizer
            final_sequences.append(text)

        return final_sequences




    @torch.no_grad()
    def generate_greedy(self, images, gt_keywords):
        device = self.device
        batch_size = images.size(0)

        # Get the BOS and EOS token IDs from the GPT-2 tokenizer
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # Initialize sequences with <BOS> token
        sequences = torch.full((batch_size, 1), bos_id, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(self.max_gen):
            logits, _ = self(images, sequences, gt_keywords)  # Forward pass
            logits = logits[:, -1, :] / self.temperature  # Get logits for the last token
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)  # Shape: (batch_size, 1)

            # Append the predicted token
            sequences = torch.cat((sequences, next_token), dim=1)

            # Stop generation if EOS is reached
            finished |= (sequences[:, -1] == eos_id)
            if finished.all():
                break

        # Decode sequences
        final_sequences = [self.tokenizer.decode(seq.tolist(), skip_special_tokens=True) for seq in sequences]

        return final_sequences  # Return decoded sequences


    

    def configure_optimizer(self,args):
        param_dict = {pn:p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]

        optim_group = [
            {'params':nodecay_params,'weight_decay':0.0},
            {'params':decay_params,'weight_decay':args.weight_decay}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'Num decay params: {len(decay_params)} with {num_decay_params} params')
        print(f'Num nodecay params: {len(nodecay_params)} with {num_nodecay_params} params')
        optimizer = torch.optim.AdamW(optim_group,lr=args.lr,betas=(0.9,0.95),eps=1e-8)
        return optimizer



            













