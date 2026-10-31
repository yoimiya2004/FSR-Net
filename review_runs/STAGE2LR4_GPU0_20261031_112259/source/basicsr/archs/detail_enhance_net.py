import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LightweightNAFBlock(nn.Module):
    """NAFNet-style lightweight restoration block without batch normalization."""

    def __init__(self, channels, dw_expand=2, ffn_expand=2):
        super(LightweightNAFBlock, self).__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.project_in = nn.Conv2d(channels, dw_channels * 2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            dw_channels * 2, dw_channels * 2, kernel_size=3, padding=1,
            groups=dw_channels * 2, bias=True)
        self.sg = SimpleGate()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels, dw_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.project_out = nn.Conv2d(dw_channels, channels, kernel_size=1, bias=True)

        self.norm2 = LayerNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_channels * 2, kernel_size=1, bias=True),
            SimpleGate(),
            nn.Conv2d(ffn_channels, channels, kernel_size=1, bias=True),
        )

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.project_in(self.norm1(x))
        y = self.dwconv(y)
        y = self.sg(y)
        y = y * self.channel_attn(y)
        y = self.project_out(y)
        x = x + y * self.beta

        y = self.ffn(self.norm2(x))
        return x + y * self.gamma


class ThinHazeBranch(nn.Module):
    """High-SNR branch: conservative local detail enhancement."""

    def __init__(self, channels):
        super(ThinHazeBranch, self).__init__()
        self.body = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1,
                      groups=channels * 2, bias=True),
            SimpleGate(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x + self.body(x) * self.scale


class DenseHazeBranch(nn.Module):
    """Low-SNR branch: cheap low-resolution context compensation."""

    def __init__(self, channels):
        super(DenseHazeBranch, self).__init__()
        self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.context = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=5, padding=2,
                      groups=channels * 2, bias=True),
            SimpleGate(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.local = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        context = self.down(x)
        context = context + self.context(context)
        context = F.interpolate(context, size=x.shape[2:], mode='bilinear', align_corners=False)
        y = self.local(x) + context
        return x + y * self.scale


class SNRFusionGate(nn.Module):
    """Use SNR as a prior while allowing features to correct local routing errors."""

    def __init__(self, channels):
        super(SNRFusionGate, self).__init__()
        self.feature_gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True),
        )
        self.gate_strength = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat, snr_map):
        if snr_map is None:
            return feat.new_full((feat.size(0), 1, feat.size(2), feat.size(3)), 0.5)

        snr_map = F.interpolate(snr_map, size=feat.shape[2:], mode='bilinear', align_corners=False)
        snr_map = snr_map.clamp(0.0, 1.0)
        correction = self.feature_gate(torch.cat([feat, snr_map], dim=1))
        gate = torch.logit(snr_map, eps=1e-4) + self.gate_strength * correction
        return torch.sigmoid(gate)


class DENet(nn.Module):
    """
    SNR-guided lightweight asymmetric detail restoration network.

    The first stage already restores the haze-dominant LL band. This module only
    predicts a small spatial residual for detail preservation and local context
    compensation.
    """

    def __init__(self, gps, blocks, conv1=default_conv):
        super(DENet, self).__init__()
        self.gps = gps
        self.dim = 16
        kernel_size = 3
        trunk_blocks = max(1, blocks // 2)

        self.pre = conv1(3, self.dim, kernel_size)
        self.trunk = nn.Sequential(*[LightweightNAFBlock(self.dim) for _ in range(trunk_blocks)])

        self.thin_branch = ThinHazeBranch(self.dim)
        self.dense_branch = DenseHazeBranch(self.dim)
        self.fusion_gate = SNRFusionGate(self.dim)

        self.transition_fusion = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=True),
        )
        self.post = conv1(self.dim, 3, kernel_size)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x1, snr_map=None):
        feat = self.trunk(self.pre(x1))
        feat_thin = self.thin_branch(feat)
        feat_dense = self.dense_branch(feat)

        gate = self.fusion_gate(feat, snr_map)
        fused = gate * feat_thin + (1.0 - gate) * feat_dense

        transition = gate * (1.0 - gate)
        cross = self.transition_fusion(torch.cat([feat_thin, feat_dense], dim=1))
        fused = fused + transition * cross

        residual = self.post(fused)
        return x1 + self.res_scale * residual


if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    snr = torch.rand(1, 1, 64, 64)
    net = DENet(3, 6)
    out = net(x, snr)
    print(out.size())
