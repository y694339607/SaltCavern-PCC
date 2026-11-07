# init_stats.py — MU/配对统计（会话执行版）
# 输入：init_repeats_per_sample.csv（同目录）；输出：控制台打印 + init_summary.csv
# 方法：腔体聚类 bootstrap（50 簇重抽样 20000 次），种子 20261031
import pandas as pd, numpy as np
from scipy import stats
from pathlib import Path

rng = np.random.default_rng(20261031)
df = pd.concat([pd.read_csv(p) for p in sorted(Path('.').glob('mu_long_table_?.csv'))], ignore_index=True)
cavs = sorted(df['cav'].unique())
cav_idx = {c: df.loc[df['cav']==c, 'i'].values for c in cavs}

def boot(d, n=20000):
    b = np.empty(n)
    for k in range(n):
        idx = np.concatenate([cav_idx[cavs[j]] for j in rng.choice(len(cavs), len(cavs), replace=True)])
        b[k] = d[idx].mean()
    return d.mean(), *np.percentile(b, [2.5, 97.5])

def boot_pooled(D, n=20000):
    b = np.empty(n)
    for k in range(n):
        idx = np.concatenate([cav_idx[cavs[j]] for j in rng.choice(len(cavs), len(cavs), replace=True)])
        b[k] = np.mean([D[p][idx].mean() for p in D])
    return b.mean(), *np.percentile(b, [2.5, 97.5])

rows = []
pairs_mu = [f'MU{k}' for k in range(1,6)]
D_CD, D_EF = {}, {}
for p in pairs_mu:
    sub = df[df['pair']==p].set_index('i')
    D_CD[p] = (sub.loc[sub.arm=='BM','cd_l1_m'] - sub.loc[sub.arm=='E2','cd_l1_m']).sort_index().values
    D_EF[p] = ((sub.loc[sub.arm=='BM','err_field_abs'] - sub.loc[sub.arm=='E2','err_field_abs'])*100).sort_index().values
    rows.append(dict(scope=f'{p}', dCD_mean=D_CD[p].mean(), dCD_neg_frac=(D_CD[p]<0).mean(),
                     dV_mean=D_EF[p].mean(), dV_neg_frac=(D_EF[p]<0).mean()))
m, l, h = boot_pooled(D_CD); rows.append(dict(scope='pooled_CD', dCD_mean=m, dCD_lo=l, dCD_hi=h))
m, l, h = boot_pooled(D_EF); rows.append(dict(scope='pooled_V', dV_mean=m, dV_lo=l, dV_hi=h))

mu_arms = df[df['pair'].isin(pairs_mu)]
for col in ['cd_l1_m','err_field_abs']:
    groups = [g[col].values for _, g in mu_arms.groupby(['pair','arm'])]
    W, pv = stats.levene(*groups, center='median')
    rows.append(dict(scope=f'BF_{col}', W=W, p=pv))

hist_bm = df[(df['pair']=='HIST')&(df['arm']=='BM')].set_index('i').sort_index()
hist_e2 = df[(df['pair']=='HIST')&(df['arm']=='E2')].set_index('i').sort_index()
for pl in ['PL1','PL2','PL3']:
    arm = df[df['pair']==pl].set_index('i').sort_index()
    for rn, ref in [('BM', hist_bm), ('E2', hist_e2)]:
        d1 = ref['cd_l1_m'].values - arm['cd_l1_m'].values
        d2 = (ref['err_field_abs'].values - arm['err_field_abs'].values)*100
        m1, l1, h1 = boot(d1); m2, l2, h2 = boot(d2)
        rows.append(dict(scope=f'{pl}_vs_{rn}', dCD_mean=m1, dCD_lo=l1, dCD_hi=h1,
                         dV_mean=m2, dV_lo=l2, dV_hi=h2))

res = pd.DataFrame(rows)
res.to_csv('init_summary.csv', index=False)
print(res.to_string())
