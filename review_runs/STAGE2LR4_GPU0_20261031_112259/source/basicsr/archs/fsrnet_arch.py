"""Paper names; legacy parameter keys and default computations are preserved."""
import torch
from torch import nn
from basicsr.utils.registry import ARCH_REGISTRY
from .wavemamba_arch import WaveMamba
from .wavemamba_haze4k_ablation_arch import WaveMambaHaze4KAblation, SS2D, LFSSBlock

@ARCH_REGISTRY.register()
class FSRNet(WaveMamba):
    """Name alias for the original LM-Net3 root training configurations."""
    pass

class ZeroResponse(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

class UniformMask(nn.Module):
    def forward(self, x):
        return x.new_full((x.shape[0], 1, x.shape[2], x.shape[3]), 0.5)

@ARCH_REGISTRY.register()
class FSRNetAblation(WaveMambaHaze4KAblation):
    """Original ablation implementation; optional single-component controls.

    Default is exactly the original implementation. 'fixed_h' isolates direction
    alternation at equal scan count; it is NOT the four-direction WDMamba baseline.
    """
    def __init__(self, scan_mode='alternating', use_lci=True, mask_mode='learned', stage2_capacity=0, **kwargs):
        if scan_mode not in ('alternating', 'fixed_h', 'fixed_v'):
            raise ValueError(scan_mode)
        if mask_mode not in ('learned', 'uniform'):
            raise ValueError(mask_mode)
        if (scan_mode != 'alternating' or not use_lci) and kwargs.get('stage1_type', 'our') != 'our':
            raise ValueError('DA/LCI controls require stage1_type=our')
        if mask_mode != 'learned' and kwargs.get('stage2_type', 'mask') != 'mask':
            raise ValueError('Mask control requires stage2_type=mask')
        super().__init__(**kwargs)
        for module in (list(self.stage1.modules()) if self.stage1 is not None else []):
            if isinstance(module, SS2D) and scan_mode != 'alternating':
                module.scan_axis = 'horizontal' if scan_mode == 'fixed_h' else 'vertical'
            if isinstance(module, LFSSBlock) and not use_lci:
                module.lci = ZeroResponse()
                module.ln_2 = nn.Identity()
        if stage2_capacity:
            if stage2_capacity != 32 or kwargs.get('stage2_type', 'mask') != 'mask':
                raise ValueError('capacity control requires mask stage2 and width32')
            with torch.random.fork_rng(devices=[]):
                self.stage2.local_branch = CapacityBranch(self.stage2.local_branch, 32, 1)
                self.stage2.global_branch = CapacityBranch(self.stage2.global_branch, 32, 2)
        if mask_mode == 'uniform':
            self.stage2.mask_head = UniformMask()


class CapacityBranch(nn.Module):
    """Retain original branch and add zero-initialized spatial residual capacity."""
    def __init__(self, original, width, dilation):
        super().__init__()
        self.original = original
        channels = original.residual_head.in_channels
        self.extra = nn.Sequential(nn.Conv2d(channels, width, 3, padding=dilation, dilation=dilation), nn.GELU(), nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation), nn.GELU(), nn.Conv2d(width, 3, 1))
        nn.init.zeros_(self.extra[-1].weight)
        nn.init.zeros_(self.extra[-1].bias)
    def forward(self, x):
        return self.original(x) + self.extra(x)
