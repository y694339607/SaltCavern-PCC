# init_fingerprint.py —  4.2：任意初始化配对重复的初始化指纹核验
# 判读：同初始化两配置（BM/E2）指纹必须一致；异初始化指纹必须不同；
#       同初始化异 order_seed 两次构建指纹必须一致（order_seed 不改初始化）。
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))
sys.path.insert(0, str(ROOT / "PoinTr"))

from train_pilot import load_config, build_model  # noqa: E402

HIST_CKPT = ROOT / "validate/pilot/ckpt/E2_full_s21_o250e250_pcV2_best.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--order_seed", type=int, required=True)
ap.add_argument("--arm", required=True)
a = ap.parse_args()

cfg = load_config(a.config)
# 与 train_pilot.py L161-162 一致（先于任何模型构建与数据消耗）
torch.manual_seed(a.seed)
np.random.seed(a.seed % (2 ** 31))
# 与 train_pilot.py L206 一致：数据顺序生成器先于模型构建创建
_g = torch.Generator().manual_seed(a.order_seed)

model = build_model(cfg["model"], "vendor")
sd = model.state_dict()
h = hashlib.sha256()
n_params = 0
for k in sorted(sd.keys()):
    t = sd[k].detach().cpu().contiguous()
    n_params += int(t.numel())
    h.update(t.numpy().tobytes())
fp = h.hexdigest()

ck = torch.load(str(HIST_CKPT), map_location="cpu", weights_only=False)
hist_sd = ck["model"] if "model" in ck else ck
hist_n = sum(int(v.numel()) for v in hist_sd.values())

out = {
    "arm": a.arm, "seed": a.seed, "order_seed": a.order_seed,
    "config": a.config,
    "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
    "torch": torch.__version__,
    "fingerprint": fp, "param_count": n_params,
    "hist_param_count": hist_n, "param_count_match": bool(n_params == hist_n),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
assert n_params == hist_n, "param count mismatch, STOP"
print("FINGERPRINT_OK", a.arm, fp[:16] + "..." + fp[-16:])
