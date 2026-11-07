"""保真三指标评估（口径：W4续 §二.3/§六；数值安全：共同参考系平移后计算，避免大坐标浮点抵消）"""
import argparse, os
import numpy as np
import pandas as pd
import torch

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

def to_dev(a, ref, device):
    return torch.from_numpy((a - ref).astype(np.float32)).to(device)

def nn_dist(a, b, ref, device, chunk=4096):
    a_t, b_t = to_dev(a, ref, device), to_dev(b, ref, device)
    out = []
    for i in range(0, a_t.size(0), chunk):
        out.append(torch.cdist(a_t[i:i+chunk], b_t).min(dim=1).values.cpu().numpy())
    return np.concatenate(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--root', default=f'{_ROOT}/dataset/pc_v2')
    ap.add_argument('--split', default='val')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta = pd.read_csv(os.path.join(args.root, 'samples.csv'), encoding='utf-8-sig')
    meta = meta[meta['split'] == args.split]
    key2group = {(r['name'], int(r['level'])): r['group'] for _, r in meta.iterrows()}
    rows = []
    for f in sorted(os.listdir(args.dump)):
        if not f.endswith('.npz'): continue
        stem = f[:-4]
        name, level = stem.rsplit('_L', 1); level = int(level)
        d = np.load(os.path.join(args.dump, f))
        fine = d['fine'].astype(np.float64); gt = d['gt'].astype(np.float64)
        obs = np.load(os.path.join(args.root, args.split, 'partial', f'{stem}.npy')).astype(np.float64)
        ref = obs.mean(axis=0)  # 共同参考系
        o2f = float(nn_dist(obs, fine, ref, device).mean())
        f2o = float(nn_dist(fine, obs, ref, device).mean())
        zmax = gt[:, 2].max()
        sel = gt[(gt[:, 2] > zmax - 60.0)]
        r_gt = np.sqrt(sel[:, 0]**2 + sel[:, 1]**2)
        keep = r_gt > 10.0
        sel, r_gt = sel[keep], r_gt[keep]
        roof_n = int(sel.shape[0]); roof = np.nan
        if roof_n >= 10:
            sel_t, fine_t = to_dev(sel, ref, device), to_dev(fine, ref, device)
            j = torch.cdist(sel_t, fine_t).argmin(dim=1).cpu().numpy()
            r_fine = np.sqrt(fine[j][:, 0]**2 + fine[j][:, 1]**2)
            roof = float(np.median(r_fine - r_gt))
        rows.append(dict(sample=stem, name=name, level=level,
                         group=key2group.get((name, level), 'NA'),
                         obs2fine=o2f, fine2obs=f2o, roof_bias=roof, roof_n=roof_n))
    df = pd.DataFrame(rows)
    agg = df.groupby('level').agg(n=('sample','count'), obs2fine_mean=('obs2fine','mean'),
        fine2obs_mean=('fine2obs','mean'), roof_bias_median=('roof_bias','median'),
        roof_n_sum=('roof_n','sum'), roof_n_na=('roof_bias', lambda s: int(s.isna().sum()))).reset_index()
    agg.to_csv(args.csv, index=False, encoding='utf-8-sig')
    print(agg.to_string(index=False)); print('FIDELITY_EVAL_DONE')

if __name__ == '__main__':
    main()
