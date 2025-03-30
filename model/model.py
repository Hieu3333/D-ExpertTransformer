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
        out = torch.matmul(att,v) #(B,nh,T,T) @ (B,nh,T,head_size) -> (B,nh,T,head_size)
        out = out.transpose(1,2).contiguous().view(B,T,self.hidden_size)
        out = self.out_proj(out)
        out = self.dropout(out)
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
        self.c_fc = nn.Linear(args.hidden_size,4*args.hidden_size,bias=args.bias)
        self.gelu = nn.ReLU()
        self.c_proj = nn.Linear(4*args.hidden_size,args.hidden_size,bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self,x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class TransfusionEncoder(nn.Module):
    def __init__(self,args):
        super(TransfusionEncoder,self).__init__()
        self.attn = DiffMultiHeadedAttention(args,depth=0,mask=False)
        self.vf_proj = nn.Linear(args.encoder_size, args.hidden_size)
        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.mlp = MLP(args)
        self.ln2 = nn.LayerNorm(args.hidden_size)
    
    def forward(self,visual_features,x):
        vf = self.vf_proj(visual_features)
        vf = vf + self.ln1(self.attn(vf,x,x))
        vf = vf + self.ln2(self.mlp(vf))
        return vf
    



class LanguageDecoderLayer(nn.Module):
    def __init__(self,args,depth):
        super(LanguageDecoderLayer,self).__init__()
        self.decoder_attn = DiffMultiHeadedAttention(args,depth=depth,mask=True)
        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.ln2 = nn.LayerNorm(args.hidden_size)
        self.ln3 = nn.LayerNorm(args.hidden_size)
        self.encoder_decoder = DiffMultiHeadedAttention(args,depth=depth,mask=False)
        self.mlp = MLP(args)

    def forward(self,encoder_feature,x): 
        x = x+ self.ln1(self.decoder_attn(x,x,x))
        x = x+ self.ln2(self.encoder_decoder(x,encoder_feature,encoder_feature))
        x = x + self.ln3(self.mlp(x))
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
        
        self.visual_extractor = EfficientNet()
        self.gca = GuidedContextAttention(args)
        self.language_encoder = DiffMultiHeadedAttention(args,depth=0,mask=False)
        self.fuser = TransfusionEncoder(args)
        # self.mlp_classifier = Classifier(args)
        self.contextual_decoder = nn.ModuleList([LanguageDecoderLayer(args,depth=depth) for depth in range(args.num_layers)])
        self.lm_head = nn.Linear(args.hidden_size,args.vocab_size, bias=False)
        self.keywords = keywords
        self.device = args.device
        self.beam_width = args.beam_width
        #Weight tying
        self.We.weight = self.lm_head.weight

    
    def forward(self,images,tokens,gt_keyword_tokens, targets = None, target_keywords=None):
        #keywords is a list of un-tokenized keywords
        #target_keywords are hot_encoding of true keywords
        B,T = tokens.shape
        device = tokens.device

        visual_features = self.visual_extractor(images) #(B,D,H,W)
        visual_features = self.gca(visual_features) #(B,H*W,D)
        # probs = probs.mean(dim=1) 
        # keywords_list = self.extract_keywords(probs,self.keywords,self.threshold)
        # keyword_tokens = self.encode_keywords(keywords_list,self.tokenizer)
        
        # keyword_tokens = keyword_tokens.to(device)

        # print('keyword_tokens max:', keyword_tokens.max().item())
        # print('vocab_size:', self.We.num_embeddings

        keyword_emb = self.We(gt_keyword_tokens) #B,max_len,hidden_size
        keyword_emb = self.language_encoder(keyword_emb,keyword_emb,keyword_emb)

        encoder_features = self.fuser(visual_features,keyword_emb)
        pos = torch.arange(0,T,dtype=torch.long,device=device)
        tok_emb = self.We(tokens)
        pos_emb = self.wpe(pos)
        x = self.dropout(tok_emb+pos_emb)

        for i in range(self.num_layers):
            x = self.contextual_decoder[i](encoder_features,x)
        
        
        logits = self.lm_head(x)
        # print("logits:",logits.shape)
        # print("target:",targets.shape)
        # print("target_keywords:",target_keywords.shape)
        if targets is not None:
            # loss_ce = F.cross_entropy(logits.view(-1,logits.shape[-1]),targets.view(-1),ignore_index=-1)
            loss_ce = F.cross_entropy(logits.permute(0, 2, 1), targets, ignore_index=-1)
            loss = loss_ce
        else:
            loss = None
            loss_ce = None
        return logits, loss, loss_ce
    

    @torch.no_grad()
    def generate(self, images, gt_keywords):
        device = self.device
        batch_size = images.size(0)
        beam_width = self.beam_width

        bos_id = self.tokenizer.word2idx["<BOS>"]
        eos_id = self.tokenizer.word2idx["<EOS>"]

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
                logits, _, _ = self(images, beam_sequences[i], gt_keywords)  # (B, seq_len, vocab_size)
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
                best_seq = best_seq[:best_seq.index(eos_id) + 1]
            text = self.tokenizer.decode(best_seq)
            final_sequences.append(text)

        return final_sequences



            


    
    def encode_keywords(self, batch_keywords, tokenizer):
        encoded_batch = []
        pad_token_id = tokenizer.word2idx["<PAD>"]

        for keywords in batch_keywords:
            keyword_list = [kw.strip() for kw in keywords]
            sep_joined = " <SEP> ".join(keyword_list)
            encoded = tokenizer.encode(sep_joined)
            
            # Truncate if longer than self.max_length
            if len(encoded) > self.max_length:
                encoded = encoded[:self.max_length]
            
            # Pad if shorter than self.max_length
            padding_length = self.max_length - len(encoded)
            padded_seq = encoded + [pad_token_id] * padding_length
            
            encoded_batch.append(torch.tensor(padded_seq))
        
        # Stack all sequences into one tensor
        padded_tensor = torch.stack(encoded_batch, dim=0)
        
        return padded_tensor

    










