# pcn_train.py
# PCN 基线在 v2 数据集上的训练入口（接入修订版）。
# 协议：250 epoch、batch 32、Adam lr 1e-4、StepLR(50, 0.7)、alpha 按 train_step 10k/20k/50k ramp、
# 固定种子 20261015——轮数与种子同 E2/BM 对齐，其余保持复现版社区协议。
import sys, time
import torch
from pathlib import Path
from torch.utils.data import DataLoader

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
sys.path.insert(0, f'{_ROOT}/coding/pcn_baseline')   # 解包目录（阶段一按指南实际路径调整）
from models import PCN
from dataset_pc import PCv2
from pcn_loss import cd_l1, HAS_EXT

CKPT_DIR = ROOT/'validate/pilot/ckpt'
BEST = CKPT_DIR/'PCN_full_s21_o250e250_pcV2_best.pt'
LAST = CKPT_DIR/'PCN_full_s21_o250e250_pcV2_last.pt'
LOG = ROOT/'train/pcn_train.log'
SEED = 20261015
EPOCHS = 250
BATCH = 32
DEVICE = 'cuda'


def alpha_at(step):
    if step < 10000: return 0.01
    if step < 20000: return 0.1
    if step < 50000: return 0.5
    return 1.0


@torch.no_grad()
def validate(model, loader):
    model.eval()
    tot, n = 0.0, 0
    for obs, gt in loader:
        obs, gt = obs.to(DEVICE), gt.to(DEVICE)
        _, fine = model(obs)
        tot += float(cd_l1(fine, gt)); n += 1
    model.train()
    return tot / max(n, 1)


def main():
    torch.manual_seed(SEED)
    import numpy as np
    np.random.seed(SEED)
    print('chamfer ext:', HAS_EXT, flush=True)
    train_loader = DataLoader(PCv2('train'), batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(PCv2('val'), batch_size=8, shuffle=False, num_workers=4)
    model = PCN(16384, 1024, 4).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.7)
    best = float('inf'); step = 0; best_ep = -1
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time(); run = 0.0
        for obs, gt in train_loader:
            alpha = alpha_at(step)
            obs, gt = obs.to(DEVICE), gt.to(DEVICE)
            coarse, fine = model(obs)
            loss = cd_l1(coarse, gt) + alpha * cd_l1(fine, gt)
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss); step += 1
        sched.step()
        val = validate(model, val_loader)
        ck = {'model': model.state_dict(), 'epoch': epoch, 'val': val, 'arch': 'PCN-repro-community'}
        torch.save(ck, LAST)
        mark = ''
        if val < best:
            best = val; best_ep = epoch
            torch.save(ck, BEST); mark = ' *'
        msg = 'epoch %d/%d alpha %.2f train %.6f val %.6f best %.6f(ep%d)%s %.0fs' % (
            epoch, EPOCHS, alpha_at(step), run/len(train_loader), val, best, best_ep, mark, time.time()-t0)
        print(msg, flush=True)
        with open(LOG, 'a') as f: f.write(msg + '\n')
    print('done best %.6f ep %d' % (best, best_ep), flush=True)


if __name__ == '__main__':
    main()
