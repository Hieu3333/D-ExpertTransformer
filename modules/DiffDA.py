import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.RMSNorm import RMSNorm
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)
def get_1d_sincos_pos_embed(seq_len, dim, device):
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
    
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # shape (1, seq_len, dim)


class MLP(nn.Module):
    def __init__(self,args):
        super(MLP,self).__init__()
        
        self.c_fc = nn.Linear(args.encoder_size,4*args.encoder_size,bias=args.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*args.encoder_size,args.encoder_size,bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self,x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
      
class DiffMultiHeadedAttention(nn.Module):
    def __init__(self,args, attn_size, depth,mask=False):
        super(DiffMultiHeadedAttention,self).__init__()
        if attn_size<1000 and attn_size==49:     
            self.diff_num_heads = 7
        else:
            self.diff_num_heads = args.diff_num_heads
        self.hidden_size = attn_size
        self.diff_head_size = attn_size // self.diff_num_heads
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_rate = args.dropout
        self.mask = mask
        
        # print('attn_size:',attn_size)
        # print('heads:',self.diff_num_heads)
        assert attn_size % self.diff_num_heads == 0

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.diff_head_size//2, dtype=torch.float32).normal_(mean=0,std=0.1))

        
        self.q_proj = nn.Linear(attn_size,attn_size,bias=args.bias)
        self.k_proj = nn.Linear(attn_size,attn_size,bias=args.bias)
        self.v_proj = nn.Linear(attn_size,attn_size,bias=args.bias)
        self.rmsnorm = RMSNorm(self.diff_head_size, eps=1e-5, elementwise_affine=True)


        self.out_proj = nn.Linear(attn_size,attn_size,bias=args.bias)
        self.register_buffer('bias',torch.tril(torch.ones(args.max_gen,args.max_gen).view(1,1,args.max_gen,args.max_gen))) 

    def forward(self,query,key,value,return_attn=False):
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
        att = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(q.shape[-1]) #(B,nh,T,N)
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
        if return_attn:
            return attn
        out = torch.matmul(attn,v) #(B,nh,T,T) @ (B,nh,T,head_size) -> (B,nh,T,head_size)
        out = self.rmsnorm(out) * (1-self.lambda_init)# (B, nh, T, head_size)
        out = out.transpose(1,2).contiguous().view(B,T,self.hidden_size)
        out = self.out_proj(out) 
        out = self.dropout(out)
        return out
    
class DiffSpatialAttention(nn.Module):
    def __init__(self, args, depth, mask=False):
        super(DiffSpatialAttention, self).__init__()
        attn_size = args.encoder_size
        self.attn = DiffMultiHeadedAttention(args, attn_size, depth, mask)
        self.return_attn = args.return_attn


    def forward(self, x):
        """
        Args:
            x: [B, H*W,C]
        Returns:
            [B,H*W,C]
        """
        out = self.attn(x, x, x,self.return_attn)  # Q = K = V = [B, HW, C]
        return out

class DiffChannelAttention(nn.Module):
    def __init__(self, args, depth, mask=False):
        super(DiffChannelAttention,self).__init__()
        if args.ve_name == 'efficientnet':
            N = 144
     
        self.attn = DiffMultiHeadedAttention(args, N, depth, mask)

    def forward(self, x):
        """
        Args:
            x: [B, H* W, C]
        Returns:
            [B,H*W,C]
        """
        B, N, C = x.shape
        x_transposed = x.transpose(1, 2)  # [B, C, HW]
        out = self.attn(x_transposed, x_transposed, x_transposed)  # [B, C, HW]
        out = out.transpose(-1,-2)
        return out

class DiffDA(nn.Module):
    def __init__(self, args):
        super(DiffDA, self).__init__()
        self.wpe = nn.Embedding(144, args.encoder_size)
        self.num_layers = args.num_layers_da
        self.return_attn = args.return_attn
        self.spatial = nn.ModuleList(DiffSpatialAttention(args, depth=depth) for depth in range(args.num_layers_da))
        self.channel = nn.ModuleList(DiffChannelAttention(args, depth=depth) for depth in range(args.num_layers_da))
        self.ffwd = nn.ModuleList(MLP(args) for _ in range(args.num_layers_da))
        self.ln1 = nn.ModuleList(nn.LayerNorm(args.encoder_size) for _ in range(args.num_layers_da))
        self.ln2 = nn.ModuleList(nn.LayerNorm(args.encoder_size) for _ in range(args.num_layers_da))
        self.device = args.device

        # Learnable fusion weights (2 per layer: one for spatial, one for channel)
        self.fusion_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([1.0, 1.0])) for _ in range(self.num_layers)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.contiguous().view(B, C, -1).transpose(-1, -2)
        pos_emb = get_1d_sincos_pos_embed(seq_len=x.size(1), dim=C, device=x.device)
        x = x + pos_emb

        for i in range(self.num_layers):
            if self.return_attn:
                return self.spatial[i](x)

            # Compute spatial and channel attention in parallel
            x_spatial = self.spatial[i](x)
            x_channel = self.channel[i](x)

            # Softmax to normalize fusion weights
            weights = F.softmax(self.fusion_weights[i], dim=0)
            alpha, beta = weights[0], weights[1]

            # Fuse the outputs with learned weights
            x_fused = alpha * x_spatial + beta * x_channel

            # Residual + LayerNorm
            x = self.ln1[i](x + x_fused)

            # Feedforward
            x = self.ln2[i](x + self.ffwd[i](x))

        return x