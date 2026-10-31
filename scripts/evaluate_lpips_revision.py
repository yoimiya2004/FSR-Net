"""LPIPS on saved paired RGB images; reports per-image data and exact protocol.

PNG/JPEG evaluation is a quantized-image protocol, not float-output evaluation.
Never compare it silently against unquantized network outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import lpips
from PIL import Image

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', type=Path, required=True)
    p.add_argument('--gt', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    suffixes = {'.png','.jpg','.jpeg','.bmp'}
    pred = {f.name:f for f in a.pred.iterdir() if f.suffix.lower() in suffixes}
    gt = {f.name:f for f in a.gt.iterdir() if f.suffix.lower() in suffixes}
    if not pred or pred.keys() != gt.keys():
        raise ValueError('Require nonempty folders with exactly matching filenames')
    if a.out.exists(): raise FileExistsError(a.out)
    metric = lpips.LPIPS(net='alex', version='0.1').eval().to(a.device)
    rows=[]
    for name in sorted(pred):
        arrays=[np.asarray(Image.open(f).convert('RGB'), dtype=np.float32)/255.0 for f in (pred[name],gt[name])]
        if arrays[0].shape != arrays[1].shape: raise ValueError('Shape mismatch: '+name)
        tensors=[torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(a.device)*2-1 for x in arrays]
        with torch.no_grad(): value=float(metric(*tensors).item())
        if not np.isfinite(value): raise ValueError('Nonfinite metric: '+name)
        rows.append(dict(name=name, lpips=value, pred_sha256=hashlib.sha256(pred[name].read_bytes()).hexdigest(),gt_sha256=hashlib.sha256(gt[name].read_bytes()).hexdigest()))
    result=dict(metric='LPIPS',backbone='alex',version='0.1',protocol='saved RGB images; decoded uint8/255; normalized to [-1,1]; full image; no resize/crop; per-image mean',torch_version=torch.__version__,pred=str(a.pred.resolve()),gt=str(a.gt.resolve()),count=len(rows),mean=float(np.mean([r['lpips'] for r in rows])),per_image=rows)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print('LPIPS',result['mean'],'N=',len(rows))

if __name__ == '__main__': main()
