# -*- coding: utf-8 -*-
"""Minimal dual-reference demo on one cavity (cav_021, three coverage tiers).

Reproduces the frozen closure-and-integration pipeline of the manuscript on the
three cloud types shipped in examples/:

  partial/    observed point cloud, 8192 points (three tiers L0/L1/L2)
  complete/   complete ground-truth cloud, 16384 points (three tiers)
  gt_field/   independent simulated radius field, (201, 360)

Reported quantities (manuscript Section 3.7):
  V_field   volume from the independent radius field
  V_ref     volume from the complete cloud through the same closure pipeline
  N_slice   number of integrable slices
  ID-8192   volume error of directly integrating all stored observations
  ID-1024   volume error of integrating the frozen 1024-point subsample

Usage:
    python examples/demo_dual_reference.py [--root examples]

Acceptance anchor (tier L0, default configuration):
    N_slice(ID-1024) = 200, N_slice(complete) = 479
    E(ID-1024 vs V_field) = 0.4559 %,  E(ID-1024 vs V_ref) = 0.6653 %
These values match results/nine_config_per_sample.csv, row cav=cav_021, level=0,
cfg=G1_500x72_max.
"""
import argparse
import os

import numpy as np

NS, NT = 500, 72
DTH = 2 * np.pi / NT
RW = 0.61          # wellbore radius (m)
Z_LO, Z_HI = 2.0, 60.0
CAV, NUM = 'cav_021', 21
LEVELS = (0, 1, 2)
SEED = 20260921


def wellbore(xc, yc, z0, z1):
    zs = np.arange(z0, z1, 0.5)
    th = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    T, Z = np.meshgrid(th, zs)
    return np.column_stack([xc + RW * np.cos(T).ravel(),
                            yc + RW * np.sin(T).ravel(), Z.ravel()])


def radial_areas(pts, xc, yc, zmin, H):
    """Cross-section area per vertical slice: 0.5 * integral r^2 dtheta."""
    pts = np.asarray(pts, dtype=np.float64)
    dz = H / NS
    k = np.clip(((pts[:, 2] - zmin) / dz).astype(int), 0, NS - 1)
    th = np.mod(np.arctan2(pts[:, 1] - yc, pts[:, 0] - xc), 2 * np.pi)
    rr = np.hypot(pts[:, 0] - xc, pts[:, 1] - yc)
    tb = np.clip((th / DTH).astype(int), 0, NT - 1)
    A = np.zeros((NS, NT))
    C = np.zeros((NS, NT), dtype=int)
    np.maximum.at(A, (k, tb), rr)
    np.add.at(C, (k, tb), 1)
    idx = np.arange(NT)
    out = np.full(NS, np.nan)
    for s in range(NS):
        m = C[s] > 0
        if m.sum() < 3:          # fewer than 3 valid azimuth cells: not integrable
            continue
        oi = np.where(m)[0]
        ei = np.concatenate([oi - NT, oi, oi + NT])
        ev = np.tile(A[s][oi], 3)
        order = np.argsort(ei)
        out[s] = 0.5 * float(np.trapezoid(np.interp(idx, ei[order], ev[order]) ** 2, dx=DTH))
    return out


def volume(a, zmin, H):
    zc = zmin + (np.arange(NS) + 0.5) * H / NS
    m = np.isfinite(a)
    if m.sum() < 10:             # fewer than 10 integrable slices: no volume
        return float('nan'), int(m.sum())
    return float(np.trapezoid(a[m], zc[m])), int(m.sum())


def id1024(obs, num, level):
    rng = np.random.Generator(np.random.Philox(
        np.random.SeedSequence(SEED, spawn_key=[int(num), int(level), 2])))
    return obs[rng.permutation(obs.shape[0])[:1024]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='examples', help='directory holding partial/, complete/, gt_field/')
    a = ap.parse_args()
    root = a.root
    gf = np.load(os.path.join(root, 'gt_field', '%s.npy' % CAV)).astype(np.float64)
    gf2 = np.where(gf > 0, gf, 0.0)

    print('%-5s %-11s %-11s %-6s %-6s %-11s %-11s %-11s' % (
        'tier', 'V_field(m3)', 'V_ref(m3)', 'N_ref', 'N_id', 'E_8192(%)', 'E_1024(%)', 'E_1024_vs_ref(%)'))
    for lv in LEVELS:
        gt = np.load(os.path.join(root, 'complete', '%s_L%d.npy' % (CAV, lv))).astype(np.float64)
        obs = np.load(os.path.join(root, 'partial', '%s_L%d.npy' % (CAV, lv))).astype(np.float64)
        zmin, zmax = gt[:, 2].min(), gt[:, 2].max()
        H = zmax - zmin
        xc, yc = gt[:, 0].mean(), gt[:, 1].mean()
        wb = wellbore(xc, yc, zmin + Z_LO, zmin + Z_HI)

        v_field = float(np.pi * np.mean(gf2 ** 2, axis=1).sum() * H / 200)
        v_ref, n_ref = volume(radial_areas(np.vstack([gt, wb]), xc, yc, zmin, H), zmin, H)
        v8192, _ = volume(radial_areas(np.vstack([obs, wb]), xc, yc, zmin, H), zmin, H)
        sub = id1024(obs, NUM, lv)
        v1024, n_id = volume(radial_areas(np.vstack([sub, wb]), xc, yc, zmin, H), zmin, H)

        e8192 = 100 * abs(v8192 - v_field) / v_field
        e1024 = 100 * abs(v1024 - v_field) / v_field
        e1024r = 100 * abs(v1024 - v_ref) / v_ref
        print('L%-4d %-11.0f %-11.0f %-6d %-6d %-11.4f %-11.4f %-11.4f'
              % (lv, v_field, v_ref, n_ref, n_id, e8192, e1024, e1024r))


if __name__ == '__main__':
    main()
