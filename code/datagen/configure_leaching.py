"""V2 toml 批量修订：blank 半径 0.41-0.57 ft、注水点腔底上方 25-35 m。
种子 20261008*100000+i 逐腔派生，可复现。
用法：SCC_WORK=/path/to/d34_batch python configure_leaching.py"""
import os, re, random, pathlib
BASE = pathlib.Path(os.environ.get("SCC_WORK", "work/d34_batch"))   # TOML working tree
for i in range(500):
    p = BASE / "tomls" / f"cav_{i:03d}.toml"
    t = p.read_text(encoding="utf-8")
    floor = float(re.search(r"^floor-depth = ([\d.]+)", t, re.M).group(1))
    rng = random.Random(20261008 * 100000 + i)
    rb = round(rng.uniform(0.41, 0.57), 4)
    t = re.sub(r"geometry-data\.radii = \[[^\]]*\]",
               "geometry-data.radii = [" + ", ".join([str(rb)] * 200) + "]", t)
    t = re.sub(r"brine-injection-depth = [\d.]+",
               lambda m: "brine-injection-depth = %.4f" % (floor - rng.uniform(82.0, 115.0)), t)
    p.write_text(t, encoding="utf-8")
print("ALL 500 DONE")
