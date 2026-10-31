import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.fsrnet_arch import FSRNetAblation

class ResidualContext(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c,c,5,padding=2*dilation,dilation=dilation,groups=c),nn.GELU(),nn.Conv2d(c,c*2,1),nn.GELU(),nn.Conv2d(c*2,c,1))
    def forward(self,x): return x + .1*self.body(x)

class DualInputRefiner(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.context_to_local = nn.Conv2d(48,24,1)
        nn.init.zeros_(self.context_to_local.weight); nn.init.zeros_(self.context_to_local.bias)
        self.local = nn.Sequential(nn.Conv2d(6,24,3,padding=1),nn.GELU(),ResidualContext(24,1),ResidualContext(24,1))
        self.context = nn.Sequential(nn.Conv2d(6,32,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(32,48,3,stride=2,padding=1),nn.GELU(),ResidualContext(48,1),ResidualContext(48,2),ResidualContext(48,3))
        self.local_head=nn.Conv2d(24,3,3,padding=1)
        self.context_head=nn.Conv2d(48,3,3,padding=1)
        self.mask=nn.Sequential(nn.Conv2d(72,16,1),nn.GELU(),nn.Conv2d(16,1,3,padding=1),nn.Sigmoid())
        for layer in [self.local_head,self.context_head]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self, coarse, hazy):
        base, _ = self.base(coarse)
        joined=torch.cat([hazy,coarse],1)
        local=self.local(joined);context=self.context(joined)
        context=F.interpolate(context,size=coarse.shape[-2:],mode='bilinear',align_corners=False)
        mask=self.mask(torch.cat([local,context],1))
        output=base+mask*self.local_head(local+self.context_to_local(context))+(1-mask)*self.context_head(context)
        return output,mask

@ARCH_REGISTRY.register()
class FSRNetDualInput(FSRNetAblation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stage2=DualInputRefiner(self.stage2)
    def encode_and_decode(self,input,current_iter=None):
        ll,coarse,_=self.stage1(input)
        output,mask=self.stage2(coarse,input)
        return dict(output_ll=ll,stage1=coarse,output=output,fusion_mask=mask)
