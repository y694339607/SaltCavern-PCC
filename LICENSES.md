# Licences · 许可

This repository mixes three kinds of material, each under its own terms.

本仓库包含三类材料，各自适用不同许可。

## 1. Code · 代码

MIT — see [LICENSE](LICENSE).

## 2. Data · 数据

The benchmark data (`dataset_pc_v2.zip` and the per-sample tables in `results/`) is released under
**Creative Commons Attribution 4.0 International (CC BY 4.0)**,
https://creativecommons.org/licenses/by/4.0/.
· 基准数据（`dataset_pc_v2.zip` 与 `results/` 中的逐样本表）以 **CC BY 4.0** 发布。

## 3. Third-party components · 第三方组件

| Component | Licence | Location |
|---|---|---|
| AdaPoinTr / PoinTr | Apache-2.0 (https://github.com/yuxumin/PoinTr) | not vendored; see `third_party/` |
| PCN (community re-implementation) | MIT | `code/baselines/pcn/`, see its `NOTICE` |
| Sansmic | BSD-3-Clause (https://github.com/sandialabs/sansmic) | not vendored; used to generate the geometry |

Third-party files retain their original licences and are not relicensed by this repository.
· 第三方文件保留其原始许可，不因本仓库而重新许可。
