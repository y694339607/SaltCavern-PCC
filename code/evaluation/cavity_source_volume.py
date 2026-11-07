# -*- coding: utf-8 -*-
"""从最终 500 份半径剖面复算轴对称腔体体积。"""
import os
from pathlib import Path
import io
import tarfile

import numpy as np
import pandas as pd

_ROOT = os.environ.get("SCC_ROOT", "/data")
_SUPP = os.environ.get("SCC_SUPP", "supplementary")
ROOT = Path(_ROOT)
SOURCE = ROOT / _SUPP / "source_data/radius500_v2.tar.gz"
OUTPUT = ROOT / _SUPP / "source_volume_500.csv"

rows = []
with tarfile.open(SOURCE, "r:gz") as archive:
    members = sorted(
        (m for m in archive.getmembers() if m.isfile() and m.name.endswith(".radius.csv")),
        key=lambda m: m.name,
    )
    if len(members) != 500:
        raise RuntimeError(f"最终剖面数量应为 500，实际为 {len(members)}")
    for member in members:
        raw = archive.extractfile(member).read()
        profile = pd.read_csv(io.BytesIO(raw))
        # 原始剖面的深度和半径均以 ft 存储，积分前转换为 m。
        z = profile.iloc[:, 0].to_numpy(dtype=np.float64) * 0.3048
        radius = profile.iloc[:, -1].to_numpy(dtype=np.float64) * 0.3048
        if len(z) != 201 or not np.all(np.isfinite(radius)):
            raise RuntimeError(f"剖面结构或数值异常：{member.name}")
        cell_height = abs(z[0] - z[-1]) / 200.0
        v_cell = float(np.pi * np.sum(radius ** 2) * cell_height)
        v_trapz = float(abs(np.trapezoid(np.pi * radius ** 2, z)))
        rows.append({
            "cav": Path(member.name).name.split(".")[0],
            "V_cell": v_cell,
            "V_trapz": v_trapz,
            "V_mean_cellheight": v_cell,
        })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT, index=False)
summary = result["V_cell"].describe(percentiles=[0.1, 0.5])
print(f"rows={len(result)}")
print(f"median_m3={summary['50%']:.6f}")
print(f"p10_m3={summary['10%']:.6f}")
print(f"max_m3={summary['max']:.6f}")
print(f"max_abs_cell_minus_trapz_m3={(result['V_cell'] - result['V_trapz']).abs().max():.6f}")
print("FINAL_500_VOLUME_RECOMPUTE_PASS")
