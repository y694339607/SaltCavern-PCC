# Replace unstable cavity configurations and record the substitutions.

"""替换协议 round-1：归档原件+记录+动力学参数重采样（不溶物/流量/blank/注水点）。
适用 ERR-A-12 不稳定腔与 v1/v2 双重失败腔。"""
import os, re, random, pathlib, csv
BASE = pathlib.Path(os.environ.get("SCC_WORK", "work/d34_batch"))   # TOML working tree
arch = BASE / "tomls" / "_archive" / "replaced_v2"
arch.mkdir(parents=True, exist_ok=True)
ledger = BASE / "replaced_v2_ledger.csv"
first = not ledger.exists()
with ledger.open("a", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    if first:
        w.writerow(["cav", "round", "reason", "insolubles", "inj_rate", "blank_r_ft", "orig_archived"])
    for i in [112, 217, 261, 313, 331, 398]:   # 按实际清单调整
        p = BASE / "tomls" / f"cav_{i:03d}.toml"
        orig = arch / f"cav_{i:03d}_orig.toml"
        if not orig.exists():
            orig.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            archived = "yes"
        else:
            archived = "already(round0)"
        t = p.read_text(encoding="utf-8")
        rng = random.Random(20261008 * 100000 + i + 1)   # round=1
        insol = round(rng.uniform(0.0, 0.1), 6)
        rate  = round(rng.uniform(1034.2, 29999.7), 4)
        rb    = round(rng.uniform(0.41, 0.57), 4)
        t = re.sub(r"insolubles-ratio = [\d.]+", f"insolubles-ratio = {insol}", t)
        t = re.sub(r"brine-injection-rate = [\d.]+", f"brine-injection-rate = {rate}", t)
        t = re.sub(r"geometry-data\.radii = \[[^\]]*\]",
                   "geometry-data.radii = [" + ", ".join([str(rb)] * 200) + "]", t)
        floor = float(re.search(r"^floor-depth = ([\d.]+)", t, re.M).group(1))
        t = re.sub(r"brine-injection-depth = [\d.]+",
                   lambda m: "brine-injection-depth = %.4f" % (floor - rng.uniform(82.0, 115.0)), t)
        p.write_text(t, encoding="utf-8")
        w.writerow([f"cav_{i:03d}", 1, "ERR-A-12_unstable_or_v1v2_double_failure", insol, rate, rb, archived])
        print("replaced", i, "insol=", insol, "rate=", rate, "rb=", rb)
