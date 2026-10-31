"""Validate restored paths and runtime before any training is started."""
import argparse
import importlib.util
import json
from pathlib import Path
import yaml

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--opt', required=True)
    p.add_argument('--paths-only', action='store_true')
    a = p.parse_args()
    opt = yaml.safe_load(Path(a.opt).read_text(encoding='utf-8'))
    errors, report = [], {}
    for phase, ds in opt['datasets'].items():
        paths = [Path(ds[k]) for k in ('dataroot_lq', 'dataroot_gt')]
        names = [{f.name for f in d.glob('*') if f.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')} for d in paths]
        report[phase] = dict(paths=[str(d) for d in paths], counts=list(map(len, names)))
        if not all(d.is_dir() for d in paths) or not all(names):
            errors.append(phase + ': missing/empty dataset directory')
        elif names[0] != names[1]:
            errors.append(phase + ': paired filenames differ (requires explicit pairing rule)')
    for key in ('pretrain_network_g', 'resume_state'):
        value = opt.get('path', {}).get(key)
        if value and not Path(value).is_file(): errors.append('missing ' + key)
    modules = {n: importlib.util.find_spec(n) is not None for n in ('torch','torchvision','mamba_ssm','cv2','pyiqa','einops','timm','pytorch_wavelets')}
    if not a.paths_only:
        errors.extend('missing module: '+n for n,v in modules.items() if not v)
    print(json.dumps(dict(data=report, modules=modules, errors=errors), indent=2, ensure_ascii=False))
    raise SystemExit(bool(errors))

if __name__ == '__main__': main()
