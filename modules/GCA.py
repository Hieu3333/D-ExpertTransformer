import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialContext(nn.Module):
    def __init__ (self,args):
        super(SpatialContext,self).__init__()
        self.conv1d = nn.Conv2d(in_channels=args.encoder_size,out_channels=1,kernel_size=1)
    
    def forward(self,x):
        b,d,h,w = x.shape
        y = self.conv1d(x) #(B,1,H,W)
        y = F.softmax(y.view(b,1,-1),dim=-1) #(b,1,h,w) -> (b,1,h*w) 
        x = x.view(b,d,-1) #(b,d,h*w)
        out = x * y #(b,d,h*w)
        out = out.sum(dim=-1,keepdim=True) #(b,d,1)
        return out.unsqueeze(-1)
    
class ChannelContext(nn.Module):
    def __init__(self,args):
        super(ChannelContext,self).__init__()
        self.k = args.channel_reduction
        self.D = args.encoder_size
        self.reduction_size = max(1,self.D // self.k)
        self.conv1 = nn.Conv2d(in_channels=self.D,out_channels=self.reduction_size,kernel_size=1)
        self.norm = nn.LayerNorm(self.reduction_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.reduction_size,out_channels=self.D,kernel_size=1)
    
    def forward(self,spatial_feature,context_feature):
        context_feature = self.conv1(context_feature) #(b,d/k,1,1)
        context_feature = context_feature.permute(0,2,3,1) #(b,1,1,d/k)
        context_feature = self.relu(self.norm(context_feature)) #(b,1,1,d/k)
        context_feature = context_feature.permute(0,3,1,2) #(b,d/k,1,1)
        context_feature = self.conv2(context_feature) #(b,d,1,1)
        out = spatial_feature + context_feature #(b,d,h,w) + (b,d,1,1) -> (b,d,h,w)
        return out
    
class GuidedContextAttention(nn.Module):
    def __init__(self,args):
        super(GuidedContextAttention,self).__init__()
        self.conv_q = nn.Conv2d(args.encoder_size,args.encoder_size,kernel_size=1)
        self.conv_k = nn.Conv2d(args.encoder_size,args.encoder_size,kernel_size=1)
        self.bias_qk = nn.Parameter(torch.zeros(1,args.encoder_size,1,1))
        self.relu = nn.ReLU()
        self.conv_f = nn.Conv2d(args.encoder_size,1,kernel_size=1)
        self.bias_f = nn.Parameter(torch.zeros(1,1,1,1))
        self.signoid = nn.Sigmoid()
        self.spatial_context = SpatialContext(args)
        self.channel_context = ChannelContext(args)

    def forward(self,x):
        b,d,h,w = x.shape
        query = self.conv_q(x) #(b,c,h,w)
        key = self.channel_context(x,self.spatial_context(x)) #(b,d,h,w)
        key = self.conv_k(key) #(b,c,h,w)
        qk = self.relu(query + key + self.bias_qk) #(b,c,h,w)
        qk = self.conv_f(qk) + self.bias_f #(b,1,h,w)
        qk = self.signoid(qk)
        out = qk * x
        return out.view(b,d,-1).transpose(-1,-2) #(b,h*w,d)
        

