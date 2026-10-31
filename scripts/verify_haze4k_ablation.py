#!/usr/bin/env python3
import gc
from pathlib import Path

import cv2
import torch

from basicsr.archs import build_network


VARIANTS = {
    'A1': ('our', 'none'),
    'A2': ('identity', 'mask'),
    'A3': ('wdmamba', 'mask'),
    'A4': ('our', 'wdmamba'),
    'A5': ('our', 'mask'),
}


def config(stage1, stage2):
    return dict(type='WaveMambaHaze4KAblation', in_chn=3, wf=16,
                n_l_blocks=[1, 2, 2, 4], ffn_scale=2.0,
                stage1_type=stage1, stage2_type=stage2)


def main():
    image_path = next(Path('/home/iris/data/去雾/Haze4K/test/haze').iterdir())
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Could not read {image_path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    image = image[:, :, :256, :256].cuda()

    checkpoint_dir = Path('experiments/Haze4K_ablation_verification')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name, (stage1, stage2) in VARIANTS.items():
        model = build_network(config(stage1, stage2)).cuda().train()
        x = torch.randn(1, 3, 64, 64, device='cuda', requires_grad=True)
        result = model(x)
        loss = result['output'].abs().mean()
        if result['output_ll'] is not None:
            loss = loss + result['output_ll'].abs().mean()
        loss.backward()

        checkpoint = checkpoint_dir / f'{name}_roundtrip.pth'
        torch.save(model.state_dict(), checkpoint)
        clone = build_network(config(stage1, stage2))
        clone.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)

        model.eval()
        with torch.no_grad():
            real_result = model(image)
        print(
            name,
            f'params={sum(p.numel() for p in model.parameters())}',
            f'stage1={type(model.stage1).__name__ if model.stage1 else "Identity"}',
            f'stage2={type(model.stage2).__name__ if model.stage2 else "None"}',
            f'synthetic_shape={tuple(result["output"].shape)}',
            f'real={image_path.name}:{tuple(real_result["output"].shape)}',
            'forward=ok backward=ok save_restore=ok',
        )
        del model, clone, x, result, loss, real_result
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
