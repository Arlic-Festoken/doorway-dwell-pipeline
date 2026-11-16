"""Masked multiscale temporal convolutions over ordered frame descriptors."""
import torch
from torch import nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Linear(1280,160)
        self.input=nn.Linear(160*2+7,192)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(192,192,3,padding=d,dilation=d),nn.GELU(),nn.Dropout(.1),nn.Conv1d(192,192,1)) for d in (1,2,4)])
        self.norms=nn.ModuleList([nn.LayerNorm(192) for _ in range(3)])
        self.attention=nn.Linear(192,1)
        self.output=nn.Sequential(nn.Linear(192*3,256),nn.GELU(),nn.Linear(256,1280))
        nn.init.normal_(self.output[-1].weight,std=.001);nn.init.zeros_(self.output[-1].bias)

    def forward(self,appearance,geometry,mask):
        a=self.project(appearance);delta=torch.cat([torch.zeros_like(a[:,:1]),a[:,1:]-a[:,:-1]],1)
        x=self.input(torch.cat([a,delta,geometry],-1))*mask[:,:,None]
        for conv,norm in zip(self.blocks,self.norms):x=norm(x+conv(x.transpose(1,2)).transpose(1,2))*mask[:,:,None]
        weights=self.attention(x).squeeze(-1).masked_fill(mask==0,-1e4).softmax(1)
        pooled=(x*weights[:,:,None]).sum(1)
        maximum=x.masked_fill(mask[:,:,None]==0,-1e4).max(1).values
        first=(mask>0).float().argmax(1);last=mask.shape[1]-1-(mask.flip(1)>0).float().argmax(1);batch=torch.arange(len(x),device=x.device)
        change=x[batch,last]-x[batch,first]
        residual=self.output(torch.cat([pooled,maximum,change],-1))
        base=(appearance*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        # The temporal branch corrects a pretrained appearance representation.
        # Its context states and ordered differences are learned end to end.
        return F.normalize(base+.35*residual,dim=-1)
