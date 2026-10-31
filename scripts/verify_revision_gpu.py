"""Actual GPU smoke tests; no long training, no writes to original experiments."""
import copy
import io
import json
from pathlib import Path
import torch
import torchvision
import yaml
from basicsr.archs.fsrnet_arch import FSRNet, FSRNetAblation
from basicsr.archs.wavemamba_arch import WaveMamba
from basicsr.archs.wavemamba_haze4k_ablation_arch import WaveMambaHaze4KAblation
from basicsr.models import build_model
from basicsr.data import create_dataset

root=Path(__file__).resolve().parents[1]
report={'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,'gpus':[], 'aliases':[], 'controls':[]}
torch.set_num_threads(4)
assert torch.cuda.device_count()==2, 'Expected two visible GPUs'
args=dict(in_chn=3,wf=16,n_l_blocks=[1,2,2,4],ffn_scale=2.0)
for gpu in range(2):
    device=torch.device('cuda:'+str(gpu));torch.cuda.set_device(device)
    x=torch.rand(1,3,64,64,device=device)
    net=FSRNetAblation(**args).to(device).train()
    output=net(x)['output'];loss=output.square().mean();loss.backward();torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in net.stage1.parameters())
    assert net.stage2.mask_head.proj.weight.grad.abs().sum()>0
    report['gpus'].append(dict(index=gpu,name=torch.cuda.get_device_name(gpu),forward_backward='passed',params=sum(p.numel() for p in net.parameters())))
    del net,output,loss;torch.cuda.empty_cache()
torch.cuda.set_device(0);x=torch.rand(1,3,64,64,device='cuda:0')
for old,new in [(WaveMamba,FSRNet),(WaveMambaHaze4KAblation,FSRNetAblation)]:
    legacy=old(**args).cuda().eval();renamed=new(**args).cuda().eval()
    renamed.load_state_dict(legacy.state_dict(),strict=True)
    with torch.no_grad():
        a=legacy(x);b=renamed(x)
        if isinstance(a,dict):a=a['output'];b=b['output']
    assert torch.equal(a,b),new.__name__
    report['aliases'].append(dict(old=old.__name__,new=new.__name__,strict_keys=True,identical_output=True,params=sum(p.numel() for p in renamed.parameters())))
    del legacy,renamed;torch.cuda.empty_cache()
for control in [dict(scan_mode='fixed_h'),dict(use_lci=False),dict(mask_mode='uniform'),dict(stage1_type='identity',stage2_type='mask')]:
    net=FSRNetAblation(**args,**control).cuda().train();y=net(x)['output'];y.mean().backward()
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    report['controls'].append(dict(control=control,forward_backward='passed',params=sum(p.numel() for p in net.parameters())))
    del net,y;torch.cuda.empty_cache()
opt=yaml.safe_load((root/'options/fsrnet/train_Haze4K_A5_Full.yml').read_text(encoding='utf-8'))
opt.update(is_train=True,dist=False,rank=0,world_size=1);opt['val']['metrics']=None
opt['datasets']['train'].update(phase='train',scale=1,gt_size=64)
dataset=create_dataset(opt['datasets']['train']);sample=dataset[0]
model=build_model(copy.deepcopy(opt));model.feed_data({k:sample[k].unsqueeze(0) for k in ('lq','gt')});model.optimize_parameters(1)
assert all(torch.isfinite(v).all() if isinstance(v,torch.Tensor) else __import__('math').isfinite(v) for v in model.get_current_log().values())
report['real_image_train_step']={k:float(v) for k,v in model.get_current_log().items()}
buffer=io.BytesIO();torch.save(dict(params=model.net_g.state_dict(),optimizer=model.optimizer_g.state_dict()),buffer);buffer.seek(0)
saved=torch.load(buffer,map_location='cuda:0',weights_only=False)
model.net_g.load_state_dict(saved['params'],strict=True);model.optimizer_g.load_state_dict(saved['optimizer'])
model.optimize_parameters(2);report['optimizer_save_restore_step']='passed'
checkpoint=Path('/home/hexiaodong/code/LM-Net3/experiments/Haze4K_A5_Full/models/net_g_latest.pth')
if checkpoint.exists():
    state=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    weights=state.get('params',state)
    model.net_g.load_state_dict(weights,strict=True)
    report['historical_checkpoint_strict_load']=str(checkpoint)
else:report['historical_checkpoint_strict_load']='not run: net_g_latest.pth absent'
(root/'revision/gpu_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
