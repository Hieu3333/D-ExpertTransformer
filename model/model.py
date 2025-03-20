import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from modules.visual_extractor import ResNet50

class MultiHeadedAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadedAttention,self).__init__()
        self.hidden_size = args.hidden_size
        self.encoder_size = args.encoder_size
        self.decoder_size = args.decoder_size
        self.num_heads = args.num_head
        self.head_size = self.hidden_size // self.num_heads
        self.dropout = nn.Dropout(args.dropout)
        assert self.hidden_size % self.num_heads == 0

        self.q_proj = nn.Linear(self.decoder_size,self.hidden_size)
        self.k_proj = nn.Linear(self.encoder_size, self.hidden_size)
        self.v_proj = nn.Linear(self.encoder_size,self.hidden_size)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self,encoder_feature,decoder_feature):
        B,N,_ = encoder_feature.shape
        B,T,_ = decoder_feature.shape #T is number of keywords
        q = self.q_proj(decoder_feature) #(B,T,C)
        k = self.k_proj(encoder_feature) #(B,N,C)
        v = self.v_proj(encoder_feature) #(B,N,C)

        q = q.view(B,T,self.num_heads,self.head_size).transpose(1,2) #(B,nh,T,head_size)
        k = k.view(B,N,self.num_heads,self.head_size).transpose(1,2) #(B,nh,N,head_size)
        v = v.view(B,N,self.num_heads,self.head_size).transpose(1,2) #(B,nh,N,head_size)

        att = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(self.head_size) #(B,nh,T,N)
        att = F.softmax(att,dim=-1)
        out = torch.matmul(att,v) #(B,nh,T,N) @ (B,nh,N,head_size) -> (B,nh,T,head_size)
        out = out.transpose(1,2).contiguous().view(B,T,self.hidden_size)
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self,args):
        super(MLP,self).__init__()
        self.c_fc = nn.Linear(args.hidden_size,4*args.hidden_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(args.hidden_size*4,args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self,x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class ImageKeywordFuser(nn.Module):
    def __init__(self,args):
        super(ImageKeywordFuser,self).__init__()
        self.attn = MultiHeadedAttention(args)
        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.mlp = MLP(args)
        self.ln2 = nn.LayerNorm(args.hidden_size)
    
    def forward(self,keyword_emb,x):
        x = x + self.ln1(self.attn(keyword_emb,x))
        x = x + self.ln2(self.mlp(x))
        return x
    
class Classifier(nn.Module):
    def __init__(self,args):
        super(Classifier,self).__init__()
        self.c_fc = nn.Linear(args.encoder_size,args.encoder_size)
        self.GELU = nn.GELU()
        self.c_proj = nn.Linear(args.encoder_sizer,args.keyword_vocab_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        return self.sigmoid(self.c_proj(self.GELU(self.c_fc(x))))



class ContextualTransformerDecoderLayer(nn.Module):
    def __init__(self,args):
        super(ContextualTransformerDecoderLayer,self).__init__()
        self.decoder_attn = MultiHeadedAttention(args)
        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.ln2 = nn.LayerNorm(args.hidden_size)
        self.ln3 = nn.LayerNorm(args.hidden_size)
        self.encoder_decoder = MultiHeadedAttention(args)
        self.mlp = MLP(args)

    def forward(self,encoder_feature,x):
        x = self.ln1(self.decoder_attn(x,x))
        x = self.ln2(self.encoder_decoder(encoder_feature,x))
        x = x + self.ln3(self.mlp(x))
        return x
    


class ExpertTransformer(nn.Module):
    def __init__(self,args):
        super(ExpertTransformer,self).__init__()
        self.We = nn.Embedding(args.vocab_size,args.hidden_size)
        self.wpe = nn.Embedding(args.max_length,args.hidden_size)
        self.max_length = args.max_length
        self.threshold = args.threshold
        self.num_layers = args.num_layers

        self.dropout = nn.Dropout(args.dropout)
        
        self.ve = ResNet50(args)
        self.fuser = ImageKeywordFuser(args)
        self.mlp_classifier = Classifier(args)
        self.contextual_decoder = nn.ModuleList([ContextualTransformerDecoderLayer(args) for _ in range(args.num_layer)])
        self.lm_head = nn.Linear(args.hidden_size,args.vocab_size, bias=False)

    

    
    def forward(self,images,tokens,keywords,targets = None):
        B,T = tokens.shape
        device = tokens.device

        visual_features = self.ve(images) #(B,N,encoder_size)
        probs = self.mlp_classifier(visual_features)
        keywords_list = self.extract_keywords_tokens(probs,keywords,self.threshold)
        keyword_emb = self.We(keywords_list)
        encoder_features = self.fuser(keyword_emb,visual_features)
        
        pos = torch.arange(0,T,dtype=torch.long,device=device)
        tok_emb = self.We(tokens)
        pos_emb = self.wpe(pos)
        x = self.dropout(tok_emb+pos_emb)

        for _ in self.num_layers:
            x = self.contextual_decoder[0](encoder_features,x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.shape[-1]),targets.view(-1),ignore_index=-1)
        else:
            loss = None
        return logits, loss
      


    
    def extract_keywords_tokens(probs, keyword_token_ids, threshold=0.5, pad_token_id=0):
        batch_token_lists = []
        
        for sample_probs in probs:
            # Get indices of keywords where prob > threshold
            indices = torch.nonzero(sample_probs > threshold, as_tuple=True)[0]
            
            # Get token IDs of those keywords
            selected_token_ids = [keyword_token_ids[i] for i in indices.tolist()]
            
            # Convert to tensor
            token_tensor = torch.tensor(selected_token_ids, dtype=torch.long)
            batch_token_lists.append(token_tensor)
        
        # Pad sequences
        padded_batch = pad_sequence(batch_token_lists, batch_first=True, padding_value=pad_token_id)  # Shape: (batch_size, max_seq_len)

        return padded_batch
    










