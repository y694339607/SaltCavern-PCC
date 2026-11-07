"""V2 全量批量：Pool(6) 并行、按 .summary.csv 断点续跑。
用法：SCC_WORK=/path/to/d34_batch python run_simulations.py
（需在装有求解器调用的环境中运行；本脚本只负责批量调度与断点续跑）"""
from multiprocessing import Pool
import subprocess, os, glob, time
def run_one(toml):
    name = os.path.splitext(os.path.basename(toml))[0]
    prefix = f"out_v2/{name}"
    if os.path.exists(prefix + ".summary.csv"):
        return name, "skip"
    r = subprocess.run(["sansmic", toml, "-o", prefix],
                       capture_output=True, timeout=3600)
    return name, ("ok" if r.returncode == 0 else "FAIL")
if __name__ == "__main__":
    t0 = time.time()
    tomls = sorted(glob.glob("tomls/cav_*.toml"))[:500]
    with Pool(6) as p:
        res = p.map(run_one, tomls)
    fails = [n for n, s in res if s == "FAIL"]
    print(f"done {len(res)-len(fails)}/{len(res)}, FAIL={fails}, {time.time()-t0:.0f}s")
