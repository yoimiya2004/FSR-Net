#!/usr/bin/env python3
import csv
import json
import time
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

from basicsr.archs import build_network


VARIANTS = {
    'A1': ('our', 'none'),
    'A2': ('identity', 'mask'),
    'A3': ('wdmamba', 'mask'),
    'A4': ('our', 'wdmamba'),
    'A5': ('our', 'mask'),
}


class TensorOutput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['output']


class SelectiveScanCounter:
    """Analytical Mamba cost using the VMamba convention (one FMA = one FLOP)."""

    def __init__(self, model):
        self.handles = []
        self.reset()
        for module in model.modules():
            cls = module.__class__
            if cls.__name__ == 'SS2D' and cls.__module__.endswith(('wavemamba_haze4k_ablation_arch', 'wdmamba_reference_wavemamba')):
                self.handles.append(module.register_forward_pre_hook(self._ss2d))
            elif cls.__name__ == 'Mamba' and cls.__module__.startswith('mamba_ssm'):
                self.handles.append(module.register_forward_pre_hook(self._mamba1d))

    def reset(self):
        self.x_proj = 0
        self.dt_proj = 0
        self.scan = 0
        self.calls = 0

    def _add(self, batch, length, d_inner, d_state, dt_rank, directions):
        self.calls += 1
        self.x_proj += batch * directions * length * d_inner * (dt_rank + 2 * d_state)
        self.dt_proj += batch * directions * length * dt_rank * d_inner
        scan_width = directions * d_inner
        self.scan += 9 * batch * length * scan_width * d_state
        self.scan += batch * length * scan_width

    def _ss2d(self, module, inputs):
        x = inputs[0]
        batch, height, width, _ = x.shape
        directions = int(module.x_proj_weight.shape[0])
        self._add(batch, height * width, module.d_inner, module.d_state,
                  module.dt_rank, directions)

    def _mamba1d(self, module, inputs):
        x = inputs[0]
        self._add(x.shape[0], x.shape[1], module.d_inner, module.d_state,
                  module.dt_rank, 1)

    def close(self):
        for handle in self.handles:
            handle.remove()


def build_variant(stage1_type, stage2_type):
    return build_network({
        'type': 'WaveMambaHaze4KAblation',
        'in_chn': 3,
        'wf': 16,
        'n_l_blocks': [1, 2, 2, 4],
        'ffn_scale': 2.0,
        'stage1_type': stage1_type,
        'stage2_type': stage2_type,
    })


def profile_variant(name, stage1_type, stage2_type, device):
    model = build_variant(stage1_type, stage2_type).to(device).eval()
    wrapped = TensorOutput(model)
    sample = torch.randn(1, 3, 256, 256, device=device)
    counter = SelectiveScanCounter(model)

    counter.reset()
    flops = FlopCountAnalysis(wrapped, sample)
    flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    raw = int(flops.total())
    counter.reset()
    with torch.no_grad():
        wrapped(sample)
    breakdown = {
        'x_proj': int(counter.x_proj),
        'dt_proj': int(counter.dt_proj),
        'selective_scan': int(counter.scan),
        'selective_scan_calls': int(counter.calls),
    }
    # fvcore traces einsum/linear projections, but not SelectiveScanFn itself.
    corrected = raw + counter.scan

    counter.close()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(10):
            wrapped(sample)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(30):
            wrapped(sample)
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000 / 30
    peak_bytes = torch.cuda.max_memory_allocated(device)

    result = {
        'variant': name,
        'stage1_type': stage1_type,
        'stage2_type': stage2_type,
        'params': sum(p.numel() for p in model.parameters()),
        'params_m': sum(p.numel() for p in model.parameters()) / 1e6,
        'raw_fvcore_gflops_fma1': raw / 1e9,
        'corrected_gflops_fma1': corrected / 1e9,
        'mamba_x_proj_gflops_fma1': breakdown['x_proj'] / 1e9,
        'mamba_dt_proj_gflops_fma1': breakdown['dt_proj'] / 1e9,
        'selective_scan_gflops_fma1': breakdown['selective_scan'] / 1e9,
        'selective_scan_calls': breakdown['selective_scan_calls'],
        'peak_inference_memory_mib': peak_bytes / 1024 ** 2,
        'latency_ms_per_image': latency_ms,
        'output_shape': list(wrapped(sample).shape),
    }
    del sample, wrapped, model
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the unified profiler.')
    device = torch.device('cuda:0')
    results = [profile_variant(name, *types, device) for name, types in VARIANTS.items()]
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    metadata = {
        'input_shape': [1, 3, 256, 256],
        'fma_convention': 'one fused multiply-add is counted as one FLOP/MAC',
        'corrected_formula': 'fvcore raw + VMamba selective scan (9*B*L*D*N + B*L*D)',
        'note': 'x_proj and dt_proj are recorded analytically for audit; fvcore already traces their einsum/linear operations.',
        'variants': results,
    }
    json_path = output_dir / 'Haze4K_ablation_complexity.json'
    csv_path = output_dir / 'Haze4K_ablation_complexity.csv'
    json_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
