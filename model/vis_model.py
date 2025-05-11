import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from modules.visual_extractor import ResNet50,EfficientNet, DenseNet
import sys
from collections import Counter
from modules.RMSNorm import RMSNorm
from modules.DiffDA import DiffDA
from modules.GCA import GuidedContextAttention



def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

    

class DiffMultiHeadedAttention(nn.Module):
    def __init__(self,args,depth,mask=True):
        super(DiffMultiHeadedAttention,self).__init__()
        self.hidden_size = args.hidden_size
        self.diff_num_heads = args.diff_num_heads
        self.diff_head_size = self.hidden_size // self.diff_num_heads
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_rate = args.dropout
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

        q = q.reshape(B,T,2*self.diff_num_heads,self.diff_head_size//2).transpose(1,2) #(B,T,2*heads,diff_head_size/2)
        k = k.reshape(B,N,2*self.diff_num_heads,self.diff_head_size//2).transpose(1,2) #(B,N,2*heads,diff_head_size/2)
        v = v.reshape(B,N,self.diff_num_heads,self.diff_head_size).transpose(1,2) #(B,nh,N,diff_head_size)
        
       
        assert q.shape[-1] == k.shape[-1]
        att = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(q.shape[-1]) #(B,2*nh,T,N)
        att = torch.nan_to_num(att)
        # print('att:',att.shape)
        if self.mask:
            att = att.masked_fill(
                    self.bias[:,:,:att.shape[2],:att.shape[3]] == 0, 
                    torch.finfo(att.dtype).min  # Ensures proper handling in mixed precision
                )
        att = F.softmax(att,dim=-1)
        att = att.reshape(B,self.diff_num_heads,2,T,-1)

        attn = att[:,:,0] - lambda_full * att[:,:,1] #(B,n_head,T,T)
        out = torch.matmul(attn,v) #(B,nh,T,T) @ (B,nh,T,head_size) -> (B,nh,T,head_size)
        out = self.rmsnorm(attn) * (1-self.lambda_init)# (B, nh, T, head_size)
        out = out.transpose(1,2).contiguous().view(B,T,self.hidden_size)
        out = self.out_proj(out) 
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self,args):
        super(MLP,self).__init__()
        
        self.c_fc = nn.Linear(args.hidden_size,args.fc_size,bias=args.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(args.fc_size,args.hidden_size,bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self,x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class TransfusionEncoder(nn.Module):
    def __init__(self,args,depth):
        super(TransfusionEncoder,self).__init__()
        self.attn = DiffMultiHeadedAttention(args,depth,mask=False)

        self.depth = depth
        self.dataset = args.dataset

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

        if self.depth == 0 and self.dataset == 'deepeyenet':
            vf = self.ln1(vf + self.attn(vf,x,x))
        else:
            vf = self.ln1(vf + self.attn(vf,vf,vf))
        vf = self.ln2(vf +self.mlp(vf))
        return vf
    
class VisualEncoder(nn.Module):
    def __init__(self,args):
        super(VisualEncoder,self).__init__()
        self.use_gca = args.use_gca
        self.return_attn = args.return_attn
        self.vis
        if args.ve_name == 'resnet':
            self.ve = ResNet50(args)
        elif args.ve_name == 'efficientnet':
            self.ve = EfficientNet(args)
        else:
            self.ve = DenseNet(args)
    
        if args.vis_processor == 'dual_attention':
            self.vis = DiffDA(args)
        elif args.vis_processor == 'gca':
            self.vis = GuidedContextAttention(args)
        else:
            self.vis = None
        if args.freeze_ve:
            for param in self.ve.parameters():
                param.requires_grad = False

    def forward(self,images):
        vf = self.ve(images)
        if self.vis is not None:
            vf = self.vis(vf)
            return vf
        else:
            B,C,_,_ = vf.size()
            vf = vf.view(B,C,-1)
            return vf.transpose(-2,-1)
        



class LanguageDecoderLayer(nn.Module):
    def __init__(self,args,depth):
        super(LanguageDecoderLayer,self).__init__()

        self.decoder_attn = DiffMultiHeadedAttention(args,depth,mask=True)
        self.encoder_decoder = DiffMultiHeadedAttention(args,depth,mask=False)

        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.ln2 = nn.LayerNorm(args.hidden_size)
        self.ln3 = nn.LayerNorm(args.hidden_size)
        
        self.mlp = MLP(args)

    def forward(self,encoder_feature,x): 
        x = self.ln1(x+self.decoder_attn(x,x,x))
        x = self.ln2(x+self.encoder_decoder(x,encoder_feature,encoder_feature))
        x = self.ln3(x +self.mlp(x))
        return x
    



class ExpertTransformer(nn.Module):
    def __init__(self,args,tokenizer):
        super(ExpertTransformer,self).__init__()
        self.We = nn.Embedding(args.vocab_size,args.hidden_size)
        self.wpe = nn.Embedding(args.max_gen,args.hidden_size)
        self.args = args
        self.max_length = args.max_length
        self.max_gen = args.max_gen
        self.num_layers = args.num_layers
        self.tokenizer = tokenizer
        self.delta1 = args.delta1
        self.delta2 = args.delta2
        self.topk = args.topk
        self.temperature = args.temperature


        self.dropout = nn.Dropout(args.dropout)
        
        self.visual_encoder = VisualEncoder(args)

        self.language_encoder = DiffMultiHeadedAttention(args,depth=0,mask=False)

        self.fuser = nn.ModuleList([TransfusionEncoder(args,depth=depth) for depth in range(args.num_layers)])
        self.contextual_decoder = nn.ModuleList([LanguageDecoderLayer(args,depth=depth) for depth in range(args.num_layers)])
        self.visual_contrastive_proj = nn.Linear(args.encoder_size, args.contrastive_proj_dim)
        self.text_contrastive_proj = nn.Linear(args.hidden_size,args.contrastive_proj_dim)
        self.lm_head = nn.Linear(args.hidden_size,args.vocab_size, bias=False)
        self.device = args.device
        self.beam_width = args.beam_width
        self.dataset = args.dataset
        self.use_contrastive = args.use_contrastive
        self.num_tokens = 10
        self.use_lt = args.use_learnable_tokens

        if args.use_learnable_tokens:
            self.learnable_tokens = nn.Parameter(torch.randn(args.batch_size,self.num_tokens,args.hidden_size)) #Use learnable tokens


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
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)


    
    def forward(self,images):
        #keywords is a list of un-tokenized keywords
        #target_keywords are hot_encoding of true keywords

        attn_map = self.visual_encoder(images)

            
        return attn_map




            













