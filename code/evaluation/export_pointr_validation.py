# export_pointr_validation.py — validation-split export of the PoinTr baseline (pointr_main_e250)
import sys, numpy as np, torch
from pathlib import Path
import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))
from paired_data.dataset import PairedDataset
from paired_data.transforms import denormalize
from train_pilot import load_config, build_model

class _DS(torch.utils.data.Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, index):
        taxonomy, cav, (partial, gt) = self.base[index]
        meta = self.base.get_meta(index, include_transform=True)
        return taxonomy, cav, (partial, gt), meta

ds = _DS(PairedDataset(root=f'{_ROOT}/dataset/pc_v2', split='val', input_points=1024,
                         gt_points=16384, downsample_seed=20260921))
cfg = load_config(str(ROOT/'train/cfg/pc_v2_pilot_pointr.yaml'))['model']

def load_model(tag):
    m = build_model(cfg, 'vendor').cuda()
    ck = torch.load(str(ROOT/'validate/pilot/ckpt'/f'{tag}.pt'), map_location='cuda', weights_only=False)
    sd = ck.get('model', ck.get('model_state', ck)) if isinstance(ck, dict) else ck
    m.load_state_dict(sd)
    m.eval()
    return m

models = {'GPoinTr': load_model('pointr_main_e250')}
out = ROOT/'validate/volume'
(out/'pred_GPoinTr').mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    for i in range(len(ds)):
        _, cav, (obs, gt), meta = ds[i]
        tp = meta['transform_params']
        obs = obs.unsqueeze(0).cuda()
        for tag, mdl in models.items():
            pred = mdl(obs)[-1]
            pred_phys = denormalize(pred[0].cpu().numpy(), tp).astype(np.float32)
            np.save(out/f'pred_{tag}'/f'{i:03d}.npy', pred_phys)
        if i % 25 == 0:
            print('progress', i, flush=True)
print('pointr_export done', flush=True)
