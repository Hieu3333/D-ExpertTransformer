import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from modules.visual_extractor import ResNet50
import sys



class MultiHeadedCrossAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadedCrossAttention,self).__init__()
        self.hidden_size = args.hidden_size
        self.encoder_size = args.encoder_size
        self.decoder_size = args.decoder_size
        self.num_heads = args.num_heads
        self.head_size = self.hidden_size // self.num_heads
        self.dropout = nn.Dropout(args.dropout)
        assert self.hidden_size % self.num_heads == 0

        self.q_proj = nn.Linear(self.decoder_size,self.hidden_size,bias=args.bias)
        self.k_proj = nn.Linear(self.encoder_size, self.hidden_size,bias=args.bias)
        self.v_proj = nn.Linear(self.encoder_size,self.hidden_size,bias=args.bias)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size,bias=args.bias)
    
    def forward(self,encoder_feature,decoder_feature):
        B,N,_ = encoder_feature.shape
        B,T,_ = decoder_feature.shape #T is number of keywords

        # print("decoder:",decoder_feature.shape)
        # print("encoder:",encoder_feature.shape)
    
        q = self.q_proj(decoder_feature) #(B,T,C)
        k = self.k_proj(encoder_feature) #(B,N,C)
        v = self.v_proj(encoder_feature) #(B,N,C)

        q = q.view(B,T,self.num_heads,self.head_size).transpose(1,2) #(B,nh,T,head_size)
        k = k.view(B,N,self.num_heads,self.head_size).transpose(1,2) #(B,nh,N,head_size)
        v = v.view(B,N,self.num_heads,self.head_size).transpose(1,2) #(B,nh,N,head_size)
        assert q.shape[-1] == k.shape[-1]
        att = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(q.shape[-1]) #(B,nh,T,N)
        att = F.softmax(att,dim=-1)
        out = torch.matmul(att,v) #(B,nh,T,N) @ (B,nh,N,head_size) -> (B,nh,T,head_size)
        out = out.transpose(1,2).contiguous().view(B,T,self.hidden_size)
        out = self.dropout(out)
        return out
    
class DiffMultiHeadedCrossAttention(nn.Module):
    def __init__(self,args):
        super(DiffMultiHeadedCrossAttention,self).__init__()
        self.hidden_size = args.hidden_size
        self.encoder_size = args.encoder_size
        self.decoder_size = args.decoder_size
        self.diff_num_heads = args.diff_num_heads
        self.diff_head_size = self.hidden_size // self.diff_num_heads
        self.lambda_init = args.lambda_init
        self.lambda_q1 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.dropout = nn.Dropout(args.dropout)
        assert self.hidden_size % self.diff_num_heads == 0

        self.q_proj = nn.Linear(self.decoder_size,self.hidden_size,bias=args.bias)
        self.k_proj = nn.Linear(self.encoder_size, self.hidden_size,bias=args.bias)
        self.v_proj = nn.Linear(self.encoder_size,self.hidden_size,bias=args.bias)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size,bias=args.bias)
    
    def forward(self,encoder_feature,decoder_feature):
        B,N,_ = encoder_feature.shape
        B,T,_ = decoder_feature.shape #T is number of keywords

        # print("decoder:",decoder_feature.shape)
        # print("encoder:",encoder_feature.shape)
    
        q = self.q_proj(decoder_feature) #(B,T,C)
        k = self.k_proj(encoder_feature) #(B,N,C)
        v = self.v_proj(encoder_feature) #(B,N,C)

        lambda1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_full = lambda1 - lambda2 + self.lambda_init

        q = q.reshape(B,T,2*self.diff_num_heads,self.diff_head_size//2).transpose(1,2) #(B,2*nh,T,diff_head_size/2)
        k = k.reshape(B,N,2*self.diff_num_heads,self.diff_head_size//2).transpose(1,2) #(B,2*nh,N,diff_head_size/2)
        v = v.reshape(B,N,self.diff_num_heads,self.diff_head_size).transpose(1,2) #(B,nh,N,2*diff_head_size)
        assert q.shape[-1] == k.shape[-1]
        att = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(q.shape[-1]) #(B,2*nh,T,N)
        att = F.softmax(att,dim=-1)
        att = att.reshape(B,self.diff_num_heads,2,T,-1) #(B,nh,2,T,N)
        att = att[:,:,0] - lambda_full * att[:,:,1] #(B,nh,T,N)
        out = torch.matmul(att,v) #(B,nh,T,N) @ (B,nh,N,head_size) -> (B,nh,T,head_size)
        out = out.transpose(1,2).contiguous().view(B,T,self.hidden_size)
        out = self.dropout(out)
        return out
    
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

    def forward(self,encoder_feature,x):
        B,T,_ = x.shape #T is number of keywords
        B,N,_ = encoder_feature.shape

        q = self.q_proj(x) #(B,T,C)
        k = self.k_proj(encoder_feature) 
        v = self.v_proj(encoder_feature) 
        

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
        out = self.dropout(out)
        return out

class DiffMultiHeadedAttention(nn.Module):
    def __init__(self,args,mask=True):
        super(DiffMultiHeadedAttention,self).__init__()
        self.hidden_size = args.hidden_size
        self.diff_num_heads = args.diff_num_heads
        self.diff_head_size = self.hidden_size // self.diff_num_heads
        self.dropout = nn.Dropout(args.dropout)
        self.mask = mask

        assert self.hidden_size % self.diff_num_heads == 0

        self.lambda_init = args.lambda_init
        self.lambda_q1 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.q_proj = nn.Linear(self.hidden_size,self.hidden_size,bias=args.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size,bias=args.bias)
        self.v_proj = nn.Linear(self.hidden_size,self.hidden_size,bias=args.bias)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size,bias=args.bias)
        self.register_buffer('bias',torch.tril(torch.ones(args.max_gen,args.max_gen).view(1,1,args.max_gen,args.max_gen))) 

    def forward(self,encoder_feature,x):
        B,T,_ = x.shape #T is number of keywords
        B,N,_ = encoder_feature.shape

        q = self.q_proj(x) #(B,T,C)
        k = self.k_proj(encoder_feature) 
        v = self.v_proj(encoder_feature) 
        
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
        out = out.transpose(1,2).contiguous().view(B,T,self.hidden_size)
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self,args):
        super(MLP,self).__init__()
        self.c_fc = nn.Linear(args.hidden_size,4*args.hidden_size,bias=args.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(args.hidden_size*4,args.hidden_size,bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self,x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class ImageKeywordFuser(nn.Module):
    def __init__(self,args):
        super(ImageKeywordFuser,self).__init__()
        self.attn = DiffMultiHeadedCrossAttention(args)
        self.ln_enc = nn.LayerNorm(args.encoder_size)
        self.ln_dec = nn.LayerNorm(args.hidden_size)
        self.mlp = MLP(args)
        self.ln2 = nn.LayerNorm(args.hidden_size)
    
    def forward(self,visual_features,x):
        x = x + self.attn(self.ln_enc(visual_features),self.ln_dec(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class Classifier(nn.Module):
    def __init__(self,args):
        super(Classifier,self).__init__()
        self.c_fc = nn.Linear(args.encoder_size,args.encoder_size,bias=args.bias)
        self.GELU = nn.GELU()
        self.c_proj = nn.Linear(args.encoder_size,args.keyword_vocab_size,bias=args.bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        logits = self.c_proj(self.GELU(self.c_fc(x)))
        logits = logits.mean(dim=1) #Average over N -> (B,1,encoder_size)
        probs = self.sigmoid(logits)
        probs = probs.mean(dim=1)
        return probs, logits



class ContextualTransformerDecoderLayer(nn.Module):
    def __init__(self,args):
        super(ContextualTransformerDecoderLayer,self).__init__()
        self.decoder_attn = DiffMultiHeadedAttention(args,mask=True)
        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.ln_enc = nn.LayerNorm(args.hidden_size)
        self.ln_dec = nn.LayerNorm(args.hidden_size)
        self.ln3 = nn.LayerNorm(args.hidden_size)
        self.encoder_decoder = DiffMultiHeadedAttention(args,mask=False)
        self.mlp = MLP(args)

    def forward(self,encoder_feature,x): 
        x = self.ln1(x)
        x = self.decoder_attn(x,x)
        x = self.encoder_decoder(self.ln_enc(encoder_feature),self.ln_dec(x))
        x = x+ self.mlp(self.ln3(x))
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

        self.dropout = nn.Dropout(args.dropout)
        
        self.visual_extractor = ResNet50()
        self.fuser = ImageKeywordFuser(args)
        self.mlp_classifier = Classifier(args)
        self.contextual_decoder = nn.ModuleList([ContextualTransformerDecoderLayer(args) for _ in range(args.num_layers)])
        self.lm_head = nn.Linear(args.hidden_size,args.vocab_size, bias=False)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.keywords = keywords
        self.device = args.device

        #Weight tying
        self.We.weight = self.lm_head.weight

    
    
    
    def forward(self,images,tokens,gt_keyword_tokens, targets = None, target_keywords=None):
        #keywords is a list of un-tokenized keywords
        #target_keywords are hot_encoding of true keywords
        B,T = tokens.shape
        device = tokens.device

        visual_features = self.visual_extractor(images) #(B,N,encoder_size)
        probs, classifier_logits = self.mlp_classifier(visual_features)
        # probs = probs.mean(dim=1) 
        # keywords_list = self.extract_keywords(probs,self.keywords,self.threshold)
        # keyword_tokens = self.encode_keywords(keywords_list,self.tokenizer)
        
        # keyword_tokens = keyword_tokens.to(device)

        # print('keyword_tokens max:', keyword_tokens.max().item())
        # print('vocab_size:', self.We.num_embeddings

        keyword_emb = self.We(gt_keyword_tokens) #B,max_len,hidden_size

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
            loss_bce = self.bce_loss(classifier_logits,target_keywords)
            loss = self.delta1*loss_ce + self.delta2*loss_bce
        else:
            loss = None
            loss_ce = None
        return logits, loss, loss_ce
    

    @torch.no_grad()
    def generate(self, images, gt_keywords, temperature=1.0):
        device = self.device
        batch_size = images.size(0)
        


        bos_id = self.tokenizer.word2idx["<BOS>"]
        eos_id = self.tokenizer.word2idx["<EOS>"]

        sequences = torch.full((batch_size, 1), bos_id, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, self.max_gen):
            logits, _, _ = self(images, sequences, gt_keywords)  # (batch_size, seq_len, vocab_size)
            logits = logits[:, -1, :] / temperature  # Scale logits

            # Apply softmax to convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Top-k Sampling
            if self.topk > 0:
                topk_probs, topk_indices = torch.topk(probs, self.topk, dim=-1)
                best_tokens = torch.gather(topk_indices, -1, torch.multinomial(topk_probs, 1)).squeeze(-1)

            # Append token to sequence
            sequences = torch.cat([sequences, best_tokens.unsqueeze(1)], dim=-1)

            # Check if EOS is reached
            finished |= best_tokens == eos_id
            if finished.all():
                break

        # Decode sequences
        final_sequences = []
        for seq in sequences.tolist():
            if eos_id in seq:
                seq = seq[:seq.index(eos_id) + 1]
            text = self.tokenizer.decode(seq)
            final_sequences.append(text)

        return final_sequences



    
    def extract_keywords(self,probs, keywords, threshold=0.9, pad_token_id=0):
        #probs (B,keyword_vocab_size)
        keywords_list = []
        # print(f"Threshold: {threshold}")
        # print(f"probs > threshold: {probs > threshold}")  # Should show True/False per element
        # print(f"indices: {torch.nonzero(probs > threshold, as_tuple=True)[0]}")

        
        for i in range(probs.shape[0]):
            # Get indices of keywords where prob > threshold
            indices = torch.nonzero(probs[i] > threshold, as_tuple=True)[0]
            # print("indices:",indices)
            # print("probs",probs[i].shape)
            
            # Get token IDs of those keywords
            selected_keywords = [keywords[i] for i in indices.tolist()]
            
            # Convert to tensor
            keywords_list.append(selected_keywords)

        return keywords_list
    
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

    










