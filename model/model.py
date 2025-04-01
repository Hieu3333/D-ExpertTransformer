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
        self.attn = MultiHeadedAttention(args,mask=False)
        self.depth = depth
        if depth == 0:
            self.vf_proj = nn.Linear(args.encoder_size, args.hidden_size)
        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.mlp = MLP(args)
        self.ln2 = nn.LayerNorm(args.hidden_size)
    
    def forward(self,visual_features,x):
        if self.depth == 0:
            vf = self.vf_proj(visual_features)
        else:
            vf = visual_features
        vf = self.ln1(vf + self.attn(vf,x,x))
        vf = self.ln2(vf +self.mlp(vf))
        return vf
    
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


class LanguageDecoderLayer(nn.Module):
    def __init__(self,args,depth):
        super(LanguageDecoderLayer,self).__init__()
        self.decoder_attn = MultiHeadedAttention(args,mask=True)
        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.ln2 = nn.LayerNorm(args.hidden_size)
        self.ln3 = nn.LayerNorm(args.hidden_size)
        self.encoder_decoder = MultiHeadedAttention(args,mask=False)
        self.mlp = MLP(args)

    def forward(self,encoder_feature,x): 
        x = self.ln1(x+self.decoder_attn(x,x,x))
        x = self.ln2(x+self.encoder_decoder(x,encoder_feature,encoder_feature))
        x = self.ln3(x +self.mlp(x))
        return x
    




class ExpertTransformer(nn.Module):
    def __init__(self,args,tokenizer,keywords):
        super(ExpertTransformer,self).__init__()
        self.We = nn.Embedding(args.vocab_size,args.hidden_size)
        self.wpe = nn.Embedding(args.max_gen,args.hidden_size)
        self.args = args
        self.max_length = args.max_length
        self.max_gen = args.max_gen
        self.threshold = args.threshold
        self.num_layers = args.num_layers
        self.tokenizer = tokenizer
        self.delta1 = args.delta1
        self.delta2 = args.delta2
        self.topk = args.topk
        self.temperature = args.temperature


        self.dropout = nn.Dropout(args.dropout)
        
        self.visual_encoder = VisualEncoder(args)
        self.language_encoder = MultiHeadedAttention(args,mask=False)
        self.fuser = nn.ModuleList([TransfusionEncoder(args,depth=depth) for depth in range(args.num_layers)])
        self.contextual_decoder = nn.ModuleList([LanguageDecoderLayer(args,depth=depth) for depth in range(args.num_layers)])
        self.lm_head = nn.Linear(args.hidden_size,args.vocab_size, bias=False)
        self.keywords = keywords
        self.device = args.device
        self.beam_width = args.beam_width
        #Weight tying
        self.We.weight = self.lm_head.weight
        self.apply(self.init_weights)

    def init_weights(self,module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    
    def forward(self,images,tokens,gt_keyword_tokens, targets = None):
        #keywords is a list of un-tokenized keywords
        #target_keywords are hot_encoding of true keywords
        B,T = tokens.shape
        device = tokens.device

        visual_features = self.visual_encoder(images)

        keyword_emb = self.We(gt_keyword_tokens) #B,keyword_length,hidden_size
        keyword_emb = self.language_encoder(keyword_emb,keyword_emb,keyword_emb)

        
        pos = torch.arange(0,T,dtype=torch.long,device=device)
        tok_emb = self.We(tokens)
        pos_emb = self.wpe(pos)
        x = self.dropout(tok_emb+pos_emb)

        for i in range(self.num_layers):
            if i==0:
                encoder_features = self.fuser[i](visual_features,keyword_emb)
            else:
                encoder_features = self.fuser[i](encoder_features,keyword_emb)
            x = self.contextual_decoder[i](encoder_features,x)
        
        
        logits = self.lm_head(x)
        # print("logits:",logits.shape)
        # print("target:",targets.shape)
        # print("target_keywords:",target_keywords.shape)
        if targets is not None:
            # loss_ce = F.cross_entropy(logits.view(-1,logits.shape[-1]),targets.view(-1),ignore_index=-1)
            loss_ce = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1), ignore_index=self.tokenizer.word2idx["<PAD>"])
            loss = loss_ce
        else:
            loss = None
            loss_ce = None
        return logits, loss, loss_ce
    

    @torch.no_grad()
    def generate_beam(self, images, gt_keywords):
        device = self.device
        batch_size = images.size(0)
        beam_width = self.beam_width
        max_gen = self.max_gen

        bos_id = self.tokenizer.word2idx["<BOS>"]
        eos_id = self.tokenizer.word2idx["<EOS>"]

        # Expand images and keywords for beam search
        images = images.unsqueeze(1).repeat(1, beam_width, 1, 1, 1)
        images = images.view(batch_size * beam_width, *images.shape[2:])

        gt_keywords = gt_keywords.unsqueeze(1).repeat(1, beam_width, 1)
        gt_keywords = gt_keywords.view(batch_size * beam_width, -1)

        # Initialize sequences, scores, and finished flags
        sequences = torch.full((batch_size * beam_width, 1), bos_id, dtype=torch.long, device=device)
        log_probs = torch.zeros(batch_size * beam_width, device=device)
        finished = torch.zeros(batch_size * beam_width, dtype=torch.bool, device=device)

        for t in range(max_gen):
            logits, _, _ = self(images, sequences, gt_keywords)  # (B*beam_width, seq_len, vocab_size)
            logits = logits[:, -1, :] / self.temperature  # Get logits for last token
            logits[finished] = float('-inf')  # Mask finished sequences
            probs = F.log_softmax(logits, dim=-1)  # Convert to log probabilities

            # Select top-k candidates (beam search step)
            top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)  # (B*beam_width, beam_width)
            log_probs = log_probs.unsqueeze(1) + top_probs  # Update log probability scores
            log_probs = log_probs.view(batch_size, beam_width**2)  # Reshape for ranking

            # Get the best beam_width sequences for each batch
            top_log_probs, top_beam_indices = torch.topk(log_probs, beam_width, dim=-1)  # (B, beam_width)

            # Correct beam index computation
            top_beam_indices_flat = top_beam_indices + (torch.arange(batch_size, device=device).unsqueeze(1) * beam_width**2)
            top_beam_indices_flat = top_beam_indices_flat.view(-1)

            # Correct indexing for new tokens
            new_tokens = top_indices.gather(1, top_beam_indices % beam_width).view(-1, 1)

            # Append new tokens
            sequences = torch.cat([sequences[top_beam_indices_flat], new_tokens], dim=-1)

            # Update log_probs
            log_probs = top_log_probs.view(-1)

            # Check if all sequences have reached EOS
            finished |= (sequences[:, -1] == eos_id)
            if finished.all():
                break

        # Decode sequences
        final_sequences = []
        for i in range(batch_size):
            best_seq_index = log_probs[i * beam_width:(i + 1) * beam_width].argmax()
            best_seq = sequences[i * beam_width + best_seq_index].tolist()
            if eos_id in best_seq:
                best_seq = best_seq[:best_seq.index(eos_id) + 1]
            text = self.tokenizer.decode(best_seq)
            final_sequences.append(text)

        return final_sequences





    @torch.no_grad()
    def generate_greedy(self, images, gt_keywords):
        device = self.device
        batch_size = images.size(0)

        bos_id = self.tokenizer.word2idx["<BOS>"]
        eos_id = self.tokenizer.word2idx["<EOS>"]

        # Initialize sequences with <BOS> token
        sequences = torch.full((batch_size, 1), bos_id, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(self.max_gen):  # Generate up to max_gen tokens
            logits, _, _ = self(images, sequences, gt_keywords)  # Forward pass
            logits = logits[:, -1, :] / self.temperature  # Get logits for last token
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)  # Ensure shape is (batch_size, 1)

            # Append the predicted token
            sequences = torch.cat((sequences, next_token), dim=1)

            # Stop generation if EOS is reached
            finished |= (sequences[:,-1] == eos_id)  # Fix shape mismatch
            if finished.all():
                break

        # Decode sequences
        final_sequences = [self.tokenizer.decode(seq.tolist()) for seq in sequences]
        
        return final_sequences


    

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



            













