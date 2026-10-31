import torch
import torch.nn as nn


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


class FusionMaskHead(nn.Module):
    """Predict a soft local-branch weight from the intermediate result."""

    def __init__(self):
        super(FusionMaskHead, self).__init__()
        self.proj = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, coarse):
        return torch.sigmoid(self.proj(coarse))


class LocalDetailBranch(nn.Module):
    """Full-resolution branch for local edges and textures."""

    def __init__(self, channels, conv1=default_conv):
        super(LocalDetailBranch, self).__init__()
        self.body = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.residual_head = conv1(channels, 3, kernel_size=3)

    def forward(self, x):
        feature = x + self.scale * self.body(x)
        return self.residual_head(feature)


class GlobalDetailBranch(nn.Module):
    """Large-receptive-field branch for broader contextual restoration."""

    def __init__(self, channels, conv1=default_conv):
        super(GlobalDetailBranch, self).__init__()
        self.body = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Conv2d(
                channels, channels, kernel_size=5, padding=4, dilation=2,
                groups=channels, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.residual_head = conv1(channels, 3, kernel_size=3)

    def forward(self, x):
        feature = x + self.scale * self.body(x)
        return self.residual_head(feature)


class DENet(nn.Module):
    """Mask-guided asymmetric residual refinement network."""

    def __init__(self, dim=16, conv1=default_conv):
        super(DENet, self).__init__()
        self.dim = dim
        self.pre = conv1(3, dim, kernel_size=3)

        self.mask_head = FusionMaskHead()
        self.local_branch = LocalDetailBranch(dim, conv1=conv1)
        self.global_branch = GlobalDetailBranch(dim, conv1=conv1)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, coarse):
        feat = self.pre(coarse)
        local_residual = self.local_branch(feat)
        global_residual = self.global_branch(feat)

        mask_input = coarse.detach() if self.training else coarse
        fusion_mask = self.mask_head(mask_input)

        local_result = coarse + self.res_scale * local_residual
        global_result = coarse + self.res_scale * global_residual
        output = fusion_mask * local_result + (1.0 - fusion_mask) * global_result

        return output, fusion_mask


if __name__ == "__main__":
    coarse = torch.randn(1, 3, 64, 64)
    net = DENet(dim=16)
    out, mask = net(coarse)
    print(out.size(), mask.size())
