"""missing_synth.py —  声呐扫描残缺合成器 v1.3（含单元 smoke）。

输入： 产物（扰动场 pert_XXX.npy / 对照剖面 cav_XXX.radius.csv，几何参数 tomls/{name}.toml）。
输出：给定腔号与缺失档位，生成观测掩膜、覆盖率、观测点云（8,192）与完整标签点云（16,384）。

五类缺失模式（ 指南 §3.1）：
  1 层间空洞    —— 按高度值定位测层（MS--02），层间距参数化（v1 目标 2 m）；
  2 角分辨率    —— 0.36° 步长扫描，映射到 1° 源网格的可观测列集合；
  3 沉渣/凸壁遮挡 —— 沉渣面之下埋置格 + 缩颈凸壁背声呐侧阴影（v1.3 复合判据，MS--01）；
  4 径向噪声    —— σ = 0.3 m 加性高斯噪声，仅作用于观测点输出（MS--06）；
  5 整层丢失    —— 按测层无置换随机丢弃，p 参数化（v1 名义 5%，实现为定数以控制覆盖率方差）。

易错点防御（ 指南 §五）：
  MS--01 模式 3 阴影只标记沉渣面之下与凸壁背声呐侧；smoke 输出剖面图供人工核对；
  MS--02 模式 1 与模式 5 一律按高度值（toml min/max 派生）定位层，禁止按行号；
  MS--03 抽样面积加权（权重 ∝ r），格内位置抖动均匀于该格所代表的面片内；
  MS--04 覆盖率统计内置于掩膜合成，smoke 逐模式打印；
  MS--05 随机流仅经 make_rng(20260921, [腔号, 档位, 用途]) 派生，全模块无裸 np.random；
  MS--06 噪声仅加在观测点径向坐标，标签与完整点云不加。

v1.3 模式 3 判据修正记录（2026-09-23，三轮 smoke 图驱动）：
  初版“上方最大半径”判据把腔顶冠部误判为凸壁，在腔顶下方产生假阴影带；
  “运行最小值”判据修复腔顶问题，但引入两个缺陷：inf 算术使腔顶边界行注册为凸壁、
  全局运行最小值使浅处小凹坑遮蔽全部深部；
  v1.3 复合判据：行向盒式平滑（w=5）后的场上，凸出行须同时满足
  ±2 行严格局部最小 + 上方 10 行窗口内最大半径超过其 5%+0.3 m + 边界/沉渣行排除；
  阴影自凸出行向下有限延伸 10 行（模拟有限波束遮蔽范围）。
  窗口最大值取代全局累计最大值是关键：深部行的累计最大值恒为腔体最宽处，
  会使深部任何噪声凹坑都满足凸出判据而产生弥漫伪阴影。

其他约定：
  高度一律用 toml 的 floor-depth / cavern-height 计算（IMP--01）；行序 floor→top；
  对照剖面 ft→m 读后立即换算（ERR-A-02），|r|<1e-3 ft 浮点垃圾归零（ERR-A-05）；
  正下限 R_MIN = 1e-3 m；对照剖面存在 _ts001 救回版时优先采用（ERR-A-12 处置链）；
  档位：0/1/2 ↔ 缺失率 30%/50%/70%（覆盖率目标 70%/50%/30%）；
  腔名以 perturb/split_400_100.csv 的 name 列为权威（全为 cav_NNN），group 列区分扰动/对照；
  遮挡受限（阴影致基底覆盖率低于目标）时取 spacing=0、p_loss=0 并标记 shadow_limited，
  覆盖率由物理遮挡决定，smoke 打印警告，该腔须记录并复核实测遮挡比例。

注意（已知限制，v1 记录）：批内核网格 Δz ∈ [4.9, 9.8] m 大于 v1 层间距规格 2 m，
模式 1 的实际层间距由标定在网格上量化实现，逐腔记录于参数表。
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

MASTER_SEED = 20260921
GRID_ROWS, GRID_COLS = 201, 360
N_FIELD = GRID_ROWS * GRID_COLS
R_MIN_M = 1e-3
FT2M = 0.3048
UNSTABLE_R_FT = 300.0
GARBAGE_R_FT = 1e-3
DEG = np.pi / 180.0
DTH_DEG = 360.0 / GRID_COLS  # 源网格角向步长 1°

# v1 缺失模式参数（ 指南 §3.1）
ANG_STEP_DEG = 0.36          # 模式 2：角分辨率
RADIAL_NOISE_M = 0.3         # 模式 4：径向噪声 σ
PROTRUSION_EPS_REL = 0.05    # 模式 3：凸出判据相对阈值（抗角向噪声）
PROTRUSION_EPS_ABS_M = 0.30  # 模式 3：凸出判据绝对阈值
RECESS_EPS_REL = 0.05        # 模式 3：背声呐侧退避相对阈值
RECESS_EPS_ABS_M = 0.30      # 模式 3：背声呐侧退避绝对阈值
SED_R_THRESH_M = 0.02        # 模式 3：沉渣埋置判据（ 沉渣截断格 1e-3 m 的量级裕度）
SHADOW_SMOOTH_W = 5          # 模式 3：行向盒式平滑窗（奇数），抑制角向噪声伪凸壁
SHADOW_LOCAL_K = 2           # 模式 3：局部最小邻域半径（行）
SHADOW_ABOVE_ROWS = 10       # 模式 3：凸壁判据参照的上方窗宽（行）
SHADOW_DEPTH_ROWS = 10       # 模式 3：阴影自凸出行向下的最大延伸（行）
P_LOSS_MAX = 0.60            # 模式 5：标定允许的最大整层丢失比例
SPACING_GRID_M = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0)

# 档位与点数（ 指南 §3.2、§3.3）
LEVEL_TARGET_COVERAGE = {0: 0.70, 1: 0.50, 2: 0.30}
COVERAGE_TOL = 0.02
N_COMPLETE = 16384
N_PARTIAL = 8192

TAG_GEN = 0  # 生成用随机流
TAG_CAL = 1  # 标定用随机流（与生成流隔离，避免标定消耗改变生成结果）

# ---------------------------------------------------------------- 随机流

def make_rng(master, tags):
    """种子派生流（ 规则 12； 扩展为 [腔号, 档位, 用途] 三元标签）。"""
    sq = np.random.SeedSequence(master, spawn_key=list(tags))
    return np.random.Generator(np.random.Philox(sq))

def parse_num(name):
    m = __import__('re').search(r'(\d+)', str(name))
    return int(m.group(1)) if m else None

# ---------------------------------------------------------------- 输入读取

def load_height_grid(toml_path):
    """由 toml 顶层键计算行高向量（201，行序 floor→top；海拔式，IMP--01）。"""
    import tomllib
    with open(toml_path, 'rb') as f:
        d = tomllib.load(f)
    fd = float(d['floor-depth'])
    H = float(d['cavern-height'])
    dz = H / (GRID_ROWS - 1)
    return fd + np.arange(GRID_ROWS) * dz

def load_field_perturbed(npy_path):
    """读取扰动场（201,360）float32，单位 m；正下限 R_MIN。"""
    a = np.load(npy_path, mmap_mode='r')
    assert a.shape == (GRID_ROWS, GRID_COLS) and a.dtype == np.float32, \
        f'扰动场规格不符: {a.shape} {a.dtype}'
    r = np.asarray(a, dtype=np.float64)
    assert np.isfinite(r).all() and r.min() > 0, '扰动场含非有限或非正值'
    return np.maximum(r, R_MIN_M)

def _read_profile_rft_ft(csv_path):
    """读取剖面末列（ft）：浮点垃圾归零（ERR-A-05），行数断言 201。"""
    import csv
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        rows = [row for row in csv.reader(f) if row]
    r_ft = np.array([float(row[-1]) for row in rows[1:]])
    assert r_ft.shape == (GRID_ROWS,), f'剖面行数 {r_ft.shape} ≠ {GRID_ROWS}'
    assert np.isfinite(r_ft).all(), f'{csv_path.name} 含非有限值'
    r_ft = r_ft.copy()
    r_ft[np.abs(r_ft) < GARBAGE_R_FT] = 0.0
    return r_ft

def resolve_profile_path(out_dir, name):
    """健康感知路径选择：标准名健康则用之，否则回退 _ts001 救回版（ERR-A-12 处置链）。"""
    p_std = out_dir / f'{name}.radius.csv'
    p_ts = out_dir / f'{name}_ts001.radius.csv'
    if p_std.exists():
        r_ft = _read_profile_rft_ft(p_std)
        if r_ft.max() < UNSTABLE_R_FT:
            return p_std
    if p_ts.exists():
        _read_profile_rft_ft(p_ts)  # 仅作存在性与可解析断言
        return p_ts
    raise FileNotFoundError(f'腔 {name} 无健康剖面（标准名与救回版均不可用）')

def load_field_control(csv_path):
    """读取对照剖面：ft→m 立即换算（ERR-A-02），轴对称广播到 (201,360)。"""
    r_ft = _read_profile_rft_ft(csv_path)
    assert r_ft.max() < UNSTABLE_R_FT, \
        f'{csv_path.name} max r = {r_ft.max():.1f} ft ≥ {UNSTABLE_R_FT:.0f} ft（ERR-A-12 不稳定）'
    r_m = np.maximum(r_ft * FT2M, R_MIN_M)
    return np.broadcast_to(r_m[:, None], (GRID_ROWS, GRID_COLS)).copy()

def load_split_table(base):
    """读取划分表，返回 {编号: (name, group)}（ 口径：name 列全为 cav_NNN）。"""
    import csv
    with open(base / 'perturb' / 'split_400_100.csv', newline='', encoding='utf-8-sig') as f:
        rows = [row for row in csv.reader(f) if row]
    header = [h.strip().lower() for h in rows[0]]
    ci, cg = header.index('name'), header.index('group')
    return {parse_num(r[ci]): (r[ci].strip(), r[cg].strip()) for r in rows[1:]}

def canonical_name(base, name):
    """以划分表为权威，将任意输入（cav_242 / pert_001 / 242）规范化到划分表 name。"""
    table = load_split_table(base)
    num = parse_num(name)
    if num not in table:
        raise KeyError(f'腔号 {num} 不在划分表中（有效编号 0–{max(table)}）')
    return table[num]

def load_field(base, name):
    """按腔名自动选择扰动场或对照剖面，返回 (r_m, kind)。"""
    base = Path(base)
    num = parse_num(name)
    pert = base / 'perturb' / f'pert_{num:03d}.npy'
    if pert.exists():
        return load_field_perturbed(pert), 'perturbed'
    return load_field_control(resolve_profile_path(base / 'out', name)), 'control'

# ---------------------------------------------------------------- 缺失模式

def mode1_row_mask(h, spacing):
    """层间空洞：按高度值定位测层（MS--02）。

    保留与理想层位 h0 + k·spacing 距离小于半行距的行；spacing ≤ 0 时保留全部行。
    """
    if spacing <= 0:
        return np.ones(GRID_ROWS, dtype=bool)
    h0 = h.min()
    dh = np.empty(GRID_ROWS)
    dh[:-1] = np.diff(h)
    dh[-1] = dh[-2]
    k = np.round((h - h0) / spacing)
    return np.abs(h - (h0 + k * spacing)) < 0.5 * dh

def mode2_col_mask(step_deg=ANG_STEP_DEG):
    """角分辨率：步长 step_deg 的测点角位置映射到 1° 源网格列集合。

    v1 规格 0.36° 细于源网格 1°，全部列可观测（贡献为 0，smoke 中打印确认）。
    """
    n = max(1, int(round(360.0 / step_deg)))
    cols = np.unique(np.floor(np.arange(n) * step_deg / DTH_DEG).astype(int) % GRID_COLS)
    m = np.zeros(GRID_COLS, dtype=bool)
    m[cols] = True
    return m

def _box_smooth_rows(a, w):
    """沿行向盒式平滑（边缘以首末行延拓），用于遮挡判据的噪声抑制。"""
    pad = w // 2
    ap = np.concatenate([np.tile(a[:1], (pad, 1)), a,
                         np.tile(a[-1:], (pad, 1))], axis=0)
    c = np.cumsum(np.vstack([np.zeros((1, a.shape[1])), ap]), axis=0)
    return (c[w:] - c[:-w]) / w

def mode3_shadow_mask(r, p_rel=PROTRUSION_EPS_REL, p_abs=PROTRUSION_EPS_ABS_M,
                      r_rel=RECESS_EPS_REL, r_abs=RECESS_EPS_ABS_M):
    """沉渣/凸壁遮挡掩膜（True = 无回波）。v1.3 判据（MS--01）：

    沉渣部分：r 低于阈值的埋置格，位于沉渣面之下。
    凸壁部分：对行向盒式平滑后的场 rd 判定——
      (a) 凸出行：rd[j] 为 ±K 行邻域严格局部最小，且上方 W2 行窗口内的最大半径
          超过 rd[j]·(1+p_rel)+p_abs（紧邻上方存在更宽区域，构成缩颈/凸壁；
          腔顶冠部为边界行，不构成凸壁，沉渣行同理排除）；
      (b) 背声呐侧阴影：凸出行下方 W 行内，rd[i] 超过窗口内最近凸出行的
          (1+r_rel)+r_abs 即判阴影（W 限定阴影深度延伸，模拟有限波束遮蔽）。
    阈值随局部半径缩放，与腔体尺寸自适应（IMP--06 类比）。
    """
    sed = r <= SED_R_THRESH_M
    rf = np.flipud(r)  # 翻转后索引 0 = 最浅行
    rd = _box_smooth_rows(rf, SHADOW_SMOOTH_W)
    # 上方 W2 行窗口内最大半径（严格上方；首行为 -inf，边界行天然不构成凸壁。
    # 用窗口最大值而非全局累计最大值：深部行的累计最大值恒为腔体最宽处，
    # 会使深部任何噪声凹坑都满足凸出判据而产生弥漫伪阴影）
    above_max = np.full((GRID_ROWS, GRID_COLS), -np.inf)
    for k in range(1, SHADOW_ABOVE_ROWS + 1):
        above_max[k:] = np.maximum(above_max[k:], rd[:GRID_ROWS - k])
    # 邻域严格局部最小（±K 行；平滑单调段无内部局部最小，锥形段不误判）
    nb_min = np.full_like(rd, np.inf)
    for k in range(1, SHADOW_LOCAL_K + 1):
        nb_min[k:] = np.minimum(nb_min[k:], rd[:-k])
        nb_min[:-k] = np.minimum(nb_min[:-k], rd[k:])
    is_local_min = rd < nb_min
    is_protrusion = (rd < above_max * (1.0 - p_rel) - p_abs) & is_local_min
    is_protrusion[0] = False               # 边界行（腔顶冠部）排除
    is_protrusion &= ~np.flipud(sed)       # 沉渣行排除
    # 上方 W 行内最近凸出行的最小半径
    big = np.where(is_protrusion, rd, np.inf)
    pmin_win = np.full((GRID_ROWS, GRID_COLS), np.inf)
    for k in range(1, SHADOW_DEPTH_ROWS + 1):
        pmin_win[k:] = np.minimum(pmin_win[k:], big[:-k])
    shadow_flip = rf > pmin_win * (1.0 + r_rel) + r_abs
    return sed | np.flipud(shadow_flip)

def mode5_layer_loss(row_mask, p_loss, rng, row_weight=None):
    """整层丢失：在 mode1 选中的测层中无置换随机丢弃 p_loss 比例（MS--02）。

    v1 规格名义 p = 5%，实际取值由标定逐腔确定；采用定数（无置换）而非独立
    Bernoulli，将覆盖率波动控制在单行权重（360/72360 ≈ 0.5%）以内；
    row_weight 给定后按权重配额截断，进一步吸收逐行遮挡数差异。
    """
    idx = np.where(row_mask)[0]
    lost = np.zeros(GRID_ROWS, dtype=bool)
    if len(idx) == 0:
        return lost
    order = rng.permutation(len(idx))
    if row_weight is None:
        k = min(int(round(p_loss * len(idx))), len(idx))
    else:
        w = np.maximum(np.asarray(row_weight, dtype=np.float64)[idx][order], 0.0)
        need = p_loss * w.sum()
        k = min(int(np.searchsorted(np.cumsum(w), need)) + 1, len(idx))
    lost[idx[order[:k]]] = True
    return lost

# ---------------------------------------------------------------- 档位标定与掩膜合成

def calibrate(r, h, level, cav):
    """按目标覆盖率标定 (spacing, p_loss)。

    对每个候选层间距求确定性基底覆盖率（mode1∧mode2∧¬mode3），
    在可行解（基底 ≥ 目标 + 余量）中取 p_loss 最小者（并列取层间距最接近 2 m 规格者）。
    全部确定性给定 (r, h, level, cav)，无随机消耗（随机流仅作未来扩展预留）。
    """
    target = LEVEL_TARGET_COVERAGE[level]
    rng = make_rng(MASTER_SEED, [cav, level, TAG_CAL])  # noqa: F841 预留
    cols = mode2_col_mask()
    shadow = mode3_shadow_mask(r)
    feasible = []
    for s in SPACING_GRID_M:
        rows = mode1_row_mask(h, s)
        bk = (np.outer(rows, cols) & ~shadow).sum() / N_FIELD
        if bk >= target + COVERAGE_TOL / 4:
            p_loss = float(np.clip(1.0 - target / bk, 0.0, P_LOSS_MAX))
            feasible.append((p_loss, abs(s - 2.0), s, bk))
    if not feasible:
        # 遮挡受限：即便保留全部测层，基底覆盖率仍低于目标（物理决定，参数无法补救）。
        # 取 spacing=0、p_loss=0 实现阴影受限覆盖率，并标记供上层记录与验收口径调整。
        rows = mode1_row_mask(h, 0.0)
        bk = (np.outer(rows, cols) & ~shadow).sum() / N_FIELD
        return {'spacing': 0.0, 'p_loss': 0.0, 'base_keep': float(bk),
                'shadow_limited': True}
    feasible.sort(key=lambda t: (t[0], t[1]))
    p_loss, _, s, bk = feasible[0]
    return {'spacing': float(s), 'p_loss': p_loss, 'base_keep': float(bk)}

def synth_masks(r, h, level, cav):
    """合成观测掩膜。返回 (observed, params)。覆盖率 = observed.sum()/72360。"""
    params = calibrate(r, h, level, cav)
    rng = make_rng(MASTER_SEED, [cav, level, TAG_GEN])
    rows = mode1_row_mask(h, params['spacing'])
    cols = mode2_col_mask()
    shadow = mode3_shadow_mask(r)
    row_obs = np.count_nonzero(cols) - np.count_nonzero(shadow, axis=1)  # 每行可观测格数
    lost = mode5_layer_loss(rows, params['p_loss'], rng, row_weight=np.maximum(row_obs, 0))
    observed = np.outer(rows, cols) & ~shadow & ~np.outer(lost, np.ones(GRID_COLS, dtype=bool))
    return observed, params

def coverage_steps(r, h, params):
    """逐模式累计覆盖率，供 smoke 打印（MS--04）。"""
    rows = mode1_row_mask(h, params['spacing'])
    cols = mode2_col_mask()
    shadow = mode3_shadow_mask(r)
    c1 = np.outer(rows, cols)
    c2 = c1 & ~shadow
    out = {'mode1_层间空洞': c1.sum() / N_FIELD,
           'mode3_遮挡后': c2.sum() / N_FIELD}
    return out

# ---------------------------------------------------------------- 抽样

def _bilinear_periodic(r, fr, fj):
    """在 (行浮点, 列浮点) 处取半径；列向周期 360，行向截断。"""
    fr = np.clip(fr, 0.0, GRID_ROWS - 1 - 1e-9)
    i0 = np.floor(fr).astype(int)
    frf = fr - i0
    fj = np.mod(fj, GRID_COLS)
    j0 = np.floor(fj).astype(int)
    fjf = fj - j0
    i1 = i0 + 1
    j1 = (j0 + 1) % GRID_COLS
    return (r[i0, j0] * (1 - frf) * (1 - fjf) + r[i1, j0] * frf * (1 - fjf)
            + r[i0, j1] * (1 - frf) * fjf + r[i1, j1] * frf * fjf)

def sample_points(r, h, cell_mask, n_points, rng, add_noise=False, noise_sigma=0.0):
    """面积加权抽稀 + 格内位置抖动（MS--03）。

    权重 ∝ r（dθ、dh 为常数因子）；无放回抽取 n_points 个格；
    高度向均匀抖动于该格行距内，角向均匀抖动于 1° 格内；
    半径取抖动位置处的双线性插值；add_noise 时加 σ 径向噪声（模式 4，仅观测点）。
    返回 (n,3) float32，列序 x, y, z（z 为海拔高）。
    """
    w = np.where(cell_mask, r, 0.0).ravel()
    total = w.sum()
    assert total > 0 and n_points <= int((w > 0).sum()), '可观测格数不足，无法抽取所需点数'
    idx = rng.choice(N_FIELD, size=n_points, replace=False, p=w / total)
    ii, jj = np.unravel_index(idx, (GRID_ROWS, GRID_COLS))
    n = n_points
    dh = np.empty(GRID_ROWS)
    dh[:-1] = np.diff(h)
    dh[-1] = dh[-2]
    dz = float(np.mean(dh))
    h_jit = h[ii] + (rng.random(n) - 0.5) * dh[ii]
    th_jit = (jj + rng.random(n)) * DTH_DEG  # [θ_j, θ_{j+1}) 内均匀
    fr = (h_jit - h[0]) / dz
    fj = th_jit / DTH_DEG
    rr = _bilinear_periodic(r, fr, fj)
    if add_noise:
        rr = np.maximum(rr + rng.normal(0.0, noise_sigma, n), R_MIN_M)  # MS--06
    th = th_jit * DEG
    pts = np.empty((n, 3), dtype=np.float32)
    pts[:, 0] = rr * np.cos(th)
    pts[:, 1] = rr * np.sin(th)
    pts[:, 2] = h_jit
    return pts

# ---------------------------------------------------------------- 体积核对

def volume_analytic(r, h):
    """解析体积 V = Σ π·mean_θ(r²)·dh（验收基准， 指南 §七.3）。"""
    dh = np.gradient(h)
    return float(np.sum(np.pi * np.mean(r ** 2, axis=1) * dh))

def volume_from_points(pts, h):
    """由点云估计体积：按高度分箱到 201 行，V = Σ π·mean(r²)·dh。"""
    dz = float(np.mean(np.diff(h)))
    rows = np.clip(np.round((pts[:, 2] - h[0]) / dz).astype(int), 0, GRID_ROWS - 1)
    rr = np.hypot(pts[:, 0], pts[:, 1]).astype(np.float64)
    cnt = np.bincount(rows, minlength=GRID_ROWS).astype(np.float64)
    r2 = np.bincount(rows, weights=rr ** 2, minlength=GRID_ROWS)
    r2m = np.divide(r2, cnt, out=np.zeros_like(r2), where=cnt > 0)
    return float(np.sum(np.pi * r2m * np.gradient(h)))

# ---------------------------------------------------------------- smoke

def _plot_mask_profile(r, h, shadow, out_png, title):
    """双面板掩膜剖面：左为 θ 平均剖面曲线，右为 θ–h 二维阴影图（MS--01 人工核对）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    prof = r.mean(axis=1)
    th = np.arange(GRID_COLS + 1) * DTH_DEG
    g = np.gradient(h)
    hh = np.concatenate([h - 0.5 * g, [h[-1] + 0.5 * g[-1]]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True,
                                   gridspec_kw={'width_ratios': [1, 1.6]})
    ax1.plot(prof, h, lw=1.2)
    ax1.set_xlabel('mean r over θ (m)')
    ax1.set_ylabel('h (m)')
    ax1.set_title('profile')
    mesh = ax2.pcolormesh(th, hh, shadow.astype(int), cmap='Reds', vmin=0, vmax=1)
    ax2.set_xlabel('θ (deg)')
    ax2.set_xlim(0, 360)
    ax2.set_title('shadow map (dark = shadowed cells)')
    fig.colorbar(mesh, ax=ax2, label='shadow flag', shrink=0.6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

def _plot_points_overlay(pts_p, pts_c, out_png, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_p[:, 0], pts_p[:, 1], pts_p[:, 2], s=1, c='tab:blue',
               alpha=0.5, label=f'partial {len(pts_p)}')
    ax.scatter(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], s=1, c='tab:orange',
               alpha=0.3, label=f'complete {len(pts_c)}')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('h (m)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

def run_smoke(base, name, outdir):
    """单腔三档 smoke：打印覆盖率、点数、体积偏差；落盘掩膜剖面图与点云叠加图。"""
    base = Path(base)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name, group = canonical_name(base, name)
    num = parse_num(name)
    r, kind = load_field(base, name)
    h = load_height_grid(base / 'tomls' / f'{name}.toml')
    print(f'[smoke] 腔 {name}（划分表分组 {group}；数据源 {kind}），'
          f'Δz = {np.mean(np.diff(h)):.3f} m，r ∈ [{r.min():.2f}, {r.max():.2f}] m')

    cov_det = None
    for level in sorted(LEVEL_TARGET_COVERAGE):
        observed, params = synth_masks(r, h, level, num)
        cov = observed.sum() / N_FIELD
        target = LEVEL_TARGET_COVERAGE[level]
        steps = coverage_steps(r, h, params)
        rng_p = make_rng(MASTER_SEED, [num, level, TAG_GEN])
        pts_c = sample_points(r, h, np.ones((GRID_ROWS, GRID_COLS), bool),
                              N_COMPLETE, rng_p)
        pts_p = sample_points(r, h, observed, N_PARTIAL, rng_p,
                              add_noise=True, noise_sigma=RADIAL_NOISE_M)
        V_ref = volume_analytic(r, h)
        V_pc = volume_from_points(pts_c, h)
        rel = abs(V_pc - V_ref) / V_ref
        # 确定性复核：同参数重算掩膜须逐位一致
        observed2, _ = synth_masks(r, h, level, num)
        det = hashlib.md5(observed.tobytes()).hexdigest() == \
            hashlib.md5(observed2.tobytes()).hexdigest()
        cov_det = det if cov_det is None else (cov_det and det)
        print(f'  档位 {level}（目标覆盖 {target:.0%}）：'
              f'层间距 {params["spacing"]:.1f} m，p_loss {params["p_loss"]:.3f}，'
              f'基底 {params["base_keep"]:.3f}')
        print(f'    逐模式累计覆盖率：' +
              '，'.join(f'{k} {v:.3f}' for k, v in steps.items()) +
              f'，合成后实测 {cov:.3f}（偏差 {cov - target:+.3f}）')
        print(f'    点数：partial {len(pts_p)}，complete {len(pts_c)}；'
              f'体积 解析 {V_ref:.1f} m³ / 点云 {V_pc:.1f} m³，相对偏差 {rel:.3%}')
        print(f'    确定性复核：{"一致" if det else "不一致（FAIL）"}')
        if params.get('shadow_limited'):
            print(f'    警告：遮挡受限（基底 {params["base_keep"]:.3f} < 目标 {target:.2f}），'
                  f'覆盖率由物理遮挡决定；该腔应记录为 shadow_limited 并复核实测遮挡比例')
        eff_target = min(target, params['base_keep'])
        assert abs(cov - eff_target) <= COVERAGE_TOL + 0.01, \
            f'档位 {level} 覆盖率偏差超容限'
        assert rel < 0.02, f'体积相对偏差 {rel:.3%} ≥ 2%'
        _plot_mask_profile(r, h, mode3_shadow_mask(r),
                           outdir / f'{name}_L{level}_mask_profile.png',
                           f'{name} L{level} shadow profile (MS--01 check)')
        _plot_points_overlay(pts_p, pts_c,
                             outdir / f'{name}_L{level}_points.png',
                             f'{name} L{level} partial vs complete')
    print(f'[smoke] 完成，图件输出至 {outdir.resolve()}')
    return 0 if cov_det else 1

def main():
    ap = argparse.ArgumentParser(description=' 残缺合成器 smoke（单腔三档）')
    ap.add_argument('--base', default='.', help='week1/d34_batch 目录')
    ap.add_argument('--cav', required=True, help='腔名（如 cav_242 或 pert_001）')
    ap.add_argument('--outdir', default='validate/dataset_check/smoke')
    args = ap.parse_args()
    raise SystemExit(run_smoke(args.base, args.cav, args.outdir))

if __name__ == '__main__':
    main()