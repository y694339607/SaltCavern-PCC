# SaltCavern-PCC

**盐穴声呐点云补全的配对仿真基准：结构化缺失、几何—库容解耦评价与训练稳定性核查**

论文《盐穴声呐点云补全的配对仿真基准：结构化缺失、几何—库容解耦评价与训练稳定性核查》（论文撰写中）的配套代码仓库。

---

## 项目概述

盐穴声呐测腔在垂向上逐层离散采样：层间没有直接回波，沉渣与凸壁形成条件相关的遮挡，作业异常还会造成整层丢失。这类逐层、柱面化且与几何相关的缺失与通用点云补全基准的随机删点协议不同；同时，补全方法通常以 Chamfer 距离类几何指标排序，而盐穴工程使用的库容是由闭合曲面积分决定的标量，几何指标本身不回答库容是否可用。

本仓库提供：

1. **配对仿真基准**：以开源造腔仿真软件 [Sansmic](https://github.com/sandialabs/sansmic) 生成 500 个腔体（400 个施加统计扰动、100 个轴对称对照），按四类声呐观测机制与一项角分辨率表示上限合成 70%/50%/30% 三档观测，共 **1500 对样本**（残缺存储 8192 点 / 模型输入 1024 点 / 真值与深度模型输出 16384 点）；数据按腔体隔离划分为训练/验证/测试 = 400/50/50 腔。
2. **几何—库容解耦评价协议**：独立仿真半径场体积 V_field 与同流程完整点云参考体积 V_ref 双参考、可积分切片计数 N_slice、三方可积分切片共同支撑 N_common、与模型输入同源的不补全基线（ID-1024 / ID-8192）。
3. **稳健性与稳定性量化**：主要终点（共同支撑下掩码配置相对 ID-1024 的体积差）在九种网格—算子组合、三次训练运行与按观测密度分层下方向均不变；三项掩码阴性对照误差均高于掩码配置；五组任意初始化重复中四组使两种监督配置同时几何退化，而内部归一化 Chamfer 指标未能预警。
4. **可复现交付**：固定划分清单、逐样本指标与积分、统计代码。

## 主要结果（物理米制）

总表（一次协议运行，验证/测试）：

| 方法 | 验证 CD1 (m) | 验证 V_field (%) | 验证 V_ref (%) | 测试 CD1 (m) | 测试 V_field (%) | 测试 V_ref (%) |
|---|---|---|---|---|---|---|
| 柱面最近邻插值（Cyl-NN） | 0.90 | 1.660 | 0.754 | 0.92 | 1.643 | 0.635 |
| AdaPoinTr 掩码配置（BM） | 0.90 | 1.459 | 1.628 | 0.97 | 1.587 | 1.889 |
| AdaPoinTr 全区域配置（E2） | 0.95 | 1.468 | 2.123 | 1.00 | 1.765 | 2.322 |
| PCN | 1.52 | 2.538 | 3.210 | 1.62 | 2.292 | 2.844 |
| 反距离加权（IDW） | 1.19 | 4.260 | 5.618 | 1.27 | 4.695 | 6.081 |
| PoinTr | 1.50 | 8.853 | 9.852 | 1.61 | 7.821 | 8.814 |
| 周期样条 | 1.77 | 198.399* | 202.430* | 1.60 | 8.668 | 9.836 |
| ID-1024（不补全） | 3.28 | 2.296 | — | 1.97 | 1.757 | — |
| ID-8192（不补全） | 2.56 | 1.503 | — | 1.12 | 1.197 | — |

*验证集均值受极端失效样本主导（中位数 4.723%/5.969%）。

要点（完整表述见论文）：

- **几何排序不能代表库容排序**，且库容排序随参考定义改变（如 IDW、周期样条几何误差优于或接近 PCN，测试集体积误差却分别为 PCN 的 2.0 倍与 3.8 倍）。
- **库容比较对积分覆盖敏感**：全支撑口径下掩码配置（1.587%，平均可积分 402.2 片）略优于 ID-1024（1.757%，157.1 片）；限制到共同支撑（155.6 片）后排序反转，**差值 +1.115 个百分点 [+0.882, +1.371]，50/50 腔同向**。该方向在九种网格—算子组合（+0.899 至 +1.960 pp）与验证集三次运行（+0.952/+1.282/+1.345 pp）下均不变。
- **基准稳定性是基准自身的属性**：五组任意初始化重复中四组两种配置同时几何退化（唯一收敛组掩码 V_field 6.745%，主运行 1.459%）；固定初始化下改变数据顺序引起的验证集体积误差极差与腔级区间宽度同量级；两者均不被腔级 bootstrap 区间覆盖。

## 仓库结构

```
SaltCavern-PCC/
├── configs/            # 训练配置（BM / E2 / PoinTr 各臂、固定初始化重复）
├── code/
│   ├── datagen/        # Sansmic 批量构建、剖面健康检查、缺失合成器、数据集组装与校验
│   ├── models/         # 训练入口与自有组件（数据集/损失/校验器），运行于 AdaPoinTr 骨干之上
│   ├── baselines/      # 规则化基线（Cyl-NN / 周期样条 / IDW）、PCN、不补全 identity 基线
│   ├── evaluation/     # CD1、体积管线（单/双参考、N_slice、共同支撑）、局部表面与逐格指标
│   ├── stats/          # 配对 bootstrap / Wilcoxon、主终点与稳定性分析
│   └── figures/        # 论文图件重绘脚本
├── third_party/        # PoinTr 兼容补丁与说明（PoinTr 库本体不入库）
├── docs/               # 数据集卡片、结果表、复现对照表
└── data/  checkpoints/  predictions/  results/   # 发布资产（见“数据与模型获取”）
```

## 数据与模型获取

- **代码**托管于本仓库（MIT 许可证）。
- **数据集（约 200 MB）、检查点与导出预测**随 Releases 发布（DOI 待定，接收后注册）。各压缩包附 SHA-256 指纹，可用 `code/datagen/verify_assets.py` 校验。
- 基准以记录坐标发布（径向已换算为米，垂向保留英尺数值），附沿垂向的仿射映射 z_m = 0.3048 z；体积类结论与该映射无关，距离类结论仅在映射后口径成立（论文 §3.1）。
- 测试划分曾在预注册封存协议下一次性用于主终点确认；为完整可复现性，测试数据与划分清单一并发布。

## 快速开始

```bash
# 1. 环境
pip install torch numpy scipy pyyaml

# 2. AdaPoinTr 骨干（不入库）：
#    克隆 https://github.com/yuxumin/PoinTr（Apache-2.0），按 third_party/README.md 操作；
#    适配已并入 code/models/train_pilot.py（训练入口两处解包点）

# 3. 数据：将数据包解至 data/pc_v2/（见 docs/DATASET.md）

# 4. 训练（单轮全量约 1.5 GPU 小时，RTX 4090）
python code/models/train_pilot.py --config configs/pc_v2_pilot_bm.yaml

# 5. 评价：导出预测后运行指标管线
python code/evaluation/export_adapointr_test.py            # 深度模型导出
python code/evaluation/baseline_recon.py           # 规则化基线
python code/evaluation/val_volume.py      # 体积 / N_slice（验证集）
python code/evaluation/val_common_support.py      # 共同支撑（验证集）
python code/stats/common_support_val_main.py     # 主终点（测试集）
```

多数分析脚本默认以 `/data` 为根路径；在其他环境运行时请修改各脚本顶部路径变量。脚本与论文的对照见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 复现说明

- 比较结论采用同腔配对设计；主要终点使用腔级聚类 bootstrap（10000 次，种子 20261103）与精确 Wilcoxon 符号秩检验、符号检验。
- n = 50 腔时双侧符号检验最小可达 p 值约为 1.8e-15，仅表示符号一致性，不携带效应强度。
- 大坐标点云的几何计算一律使用 float64 / 归一化坐标，避免灾难性抵消；全部几何指标以物理米制报告（z × 0.3048 换算后计算）。
- 运行间稳定性沿三个维度刻画：数据顺序（三次运行）、初始化（五组重复）、积分数值配置（九种网格—算子组合）；腔级区间仅条件于腔体抽样（论文 §3.8、§4.5）。

## 命名约定

文件、检查点与配置名沿用发布资产中的代号，以便与检查点、日志、指纹交叉引用；论文正式名对照：

| 代号 | 论文名 |
|---|---|
| `BM` / `pc_v2_pilot_bm*.yaml` | AdaPoinTr 掩码（盲区）监督 |
| `E2` | AdaPoinTr 全区域监督 |
| `ms_*` | 固定初始化数据顺序重复 |
| `B1` / `B2` / `B3`（基线脚本内） | Cyl-NN / 周期样条 / IDW |
| `pc_v2` | 基准数据集 |
| 文件名中 `s21` | 初始化种子 20260921；数据/顺序种子 20260921 / 20261001 |

## 引用

```bibtex
@misc{saltcavern_pcc_2026,
  title  = {A paired simulation benchmark for sonar salt-cavern point cloud completion: structured missingness, decoupled geometry--volume evaluation, and training-stability checks},
  author = {SaltCavern-PCC authors},
  year   = {2026},
  note   = {Manuscript in preparation; data and code DOI to be registered after acceptance}
}
```

## 许可证

代码：MIT（见 [LICENSE](LICENSE)）。数据集：CC BY 4.0。第三方组件（AdaPoinTr/PoinTr、PCN 重实现、Sansmic）保留各自原许可证，见 [third_party/README.md](third_party/README.md)。

## 声明

全部结论限于仿真基准与所报告的运行。统计扰动缓解但不消除仿真真值的轴对称性质；缺失机制参数未用现场声呐统计校准。现场声呐数据的外部验证是后续首要工作。
