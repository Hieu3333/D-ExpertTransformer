import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.RMSNorm import RMSNorm
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MLP(nn.Module):
    def __init__(self,args):
        super(MLP,self).__init__()
        
        self.c_fc = nn.Linear(args.hidden_size,args.fc_size,bias=args.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(args.fc_size,args.hidden_size,bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self,x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
      
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

        q = q.reshape(B,T,self.diff_num_heads,2,self.diff_head_size//2).transpose(1,2) #(B,nh,T,2,diff_head_size/2)
        k = k.reshape(B,N,self.diff_num_heads,2,self.diff_head_size//2).transpose(1,2) #(B,nh,N,2,diff_head_size/2)
        v = v.reshape(B,N,self.diff_num_heads,self.diff_head_size).transpose(1,2) #(B,nh,N,diff_head_size)
        
        q1, q2 = q[:,:,:,0], q[:,:,:,1]
        k1, k2 = k[:,:,:,0], k[:,:,:,1]
        att1 = F.scaled_dot_product_attention(q1,k1,v,attn_mask=None,dropout_p=self.dropout_rate,is_causal=self.mask)
        att2 = F.scaled_dot_product_attention(q2,k2,v,attn_mask=None,dropout_p=self.dropout_rate,is_causal=self.mask)
        # assert q.shape[-1] == k.shape[-1]
        # att = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(q.shape[-1]) #(B,nh,T,N)
        # # print('att:',att.shape)
        # if self.mask:
        #     att = att.masked_fill(
        #             self.bias[:,:,:att.shape[2],:att.shape[3]] == 0, 
        #             torch.finfo(att.dtype).min  # Ensures proper handling in mixed precision
        #         )
        # att = F.softmax(att,dim=-1)
        # att = att.reshape(B,self.diff_num_heads,2,T,-1)
        attn = att1 - lambda_full * att2
        # out = torch.matmul(att,v) #(B,nh,T,T) @ (B,nh,T,head_size) -> (B,nh,T,head_size)
        out = self.rmsnorm(attn) * (1-self.lambda_init)# (B, nh, T, head_size)
        out = out.transpose(1,2).contiguous().view(B,T,self.hidden_size)
        out = self.out_proj(out) 
        out = self.dropout(out)
        return out
    
class DiffSpatialAttention(nn.Module):
    def __init__(self, args, depth, mask=False):
        super(DiffSpatialAttention, self).__init__()
        self.attn = DiffMultiHeadedAttention(args, depth, mask)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        out = self.attn(x_flat, x_flat, x_flat)  # Q = K = V = [B, HW, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class DiffChannelAttention(nn.Module):
    def __init__(self, args, depth, mask=False):
        super(DiffChannelAttention,self).__init__()
        self.attn = DiffMultiHeadedAttention(args, depth, mask)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_perm = x.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]
        x_transposed = x_perm.transpose(1, 2)  # [B, C, HW]
        out = self.attn(x_transposed, x_transposed, x_transposed)  # [B, C, HW]
        out = out.view(B, C, H, W)
        return out

class DiffDA(nn.Module):
    def __init__(self,args):
        super(DiffDA,self).__init__()
        self.num_layers = args.num_layers_da
        self.spatial = nn.ModuleList(DiffSpatialAttention(args,depth=depth) for depth in range(args.num_layers_da))
        self.channel = nn.ModuleList(DiffChannelAttention(args,depth=depth) for depth in range(args.num_layers_da))
        self.ffwd = nn.ModuleList(MLP(args) for _ in range(args.num_layers_da))
        self.ln1 = nn.ModuleList(nn.LayerNorm() for _ in range(args.num_layers_da))
        self.ln2 = nn.ModuleList(nn.LayerNorm() for _ in range(args.num_layers_da))
    def forward(self,x):
        for i in range(self.num_layers):
            x =self.ln1[i](x + self.spatial[i](x))
            x =self.ln2[i](x + self.channel[i](x))
        B, C, H, W = x.shape
        x = x.view(B,C,-1).transpose(-1,-2) #(b,h*w,c)
        return x