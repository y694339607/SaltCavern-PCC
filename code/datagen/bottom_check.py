"""阶段一判定线：底部 1/4 段末列最大半径 >5 ft 即死区消除。"""
import csv, glob
for f in sorted(glob.glob("out_v2/cav_*.radius.csv")):
    rows = list(csv.reader(open(f)))
    col = [float(r[-1]) for r in rows[1:]]      # 首行=腔底，末列=末时刻
    q = len(col) // 4
    print(f, " bottom-quarter max r =", round(max(col[:q]), 2), "ft")
