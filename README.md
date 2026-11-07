# SaltCavern-PCC

**A paired simulation benchmark for sonar salt-cavern point cloud completion: structured missingness, decoupled geometry–volume evaluation, and training-stability checks**

Code accompanying the manuscript *盐穴声呐点云补全的配对仿真基准：结构化缺失、几何—库容解耦评价与训练稳定性核查* (manuscript in preparation).

---

## About

Salt-cavern sonar surveys sample discretely in the vertical direction: there is no direct echo between measurement layers, debris and protruding walls cast condition-dependent acoustic shadows, and operational faults can drop whole layers. This layered, cylindrical, geometry-dependent missingness differs from the random point-deletion protocol used by generic point-cloud-completion benchmarks. Meanwhile, completion methods are usually ranked by Chamfer-type geometric metrics, whereas the working-gas volume used in cavern engineering is a scalar obtained by integration over a closed surface; a geometric metric does not by itself answer whether the volume is usable.

This repository provides:

1. **A paired simulation benchmark.** 500 cavities generated with the open-source leaching simulator [Sansmic](https://github.com/sandialabs/sansmic) (400 with statistical perturbations, 100 axisymmetric controls), observed under three coverage tiers (L0/L1/L2, nominal retention 70%/50%/30%) synthesized from four sonar observation mechanisms plus an angular-resolution representation limit, giving **1500 paired samples** (incomplete 8192 points stored / model input 1024 points / complete and deep-model output 16384 points). Splits are cavity-isolated (train/val/test = 400/50/50 cavities).

2. **An evaluation protocol that separates geometric error from volume error.** Two reference volumes (an independent simulated radius field V_field, and V_ref obtained by running the complete cloud through the same closure-and-integration pipeline), integrable-slice counting (N_slice), three-way common support (N_common), and input-matched no-completion baselines (ID-1024 / ID-8192).

3. **Quantified robustness and stability checks.** The main endpoint (masked config vs ID-1024 on common support) is positive in all nine grid–operator combinations, in three training runs on the validation split, and under observational-density stratification; three placebo controls all exceed the masked configuration; four of five arbitrary-initialization repeats degrade both supervision configurations geometrically without warning from the internal normalized Chamfer metric.

4. **Reproducible delivery.** Fixed split lists (`data/splits/samples.csv`), the per-sample metrics behind every table (`results/`), and the integration and statistics code in this repository.

## Main results (physical metric units)

Overall table (one protocol run, val/test):

| Method | Val CD1 (m) | Val V_field (%) | Val V_ref (%) | Test CD1 (m) | Test V_field (%) | Test V_ref (%) |
|---|---|---|---|---|---|---|
| Cylindrical nearest-neighbor (Cyl-NN) | 0.90 | 1.660 | 0.754 | 0.92 | 1.643 | 0.635 |
| AdaPoinTr, masked supervision (BM) | 0.90 | 1.459 | 1.628 | 0.97 | 1.587 | 1.889 |
| AdaPoinTr, full-region supervision (E2) | 0.95 | 1.468 | 2.123 | 1.00 | 1.765 | 2.322 |
| PCN | 1.52 | 2.538 | 3.210 | 1.62 | 2.292 | 2.844 |
| IDW | 1.19 | 4.260 | 5.618 | 1.27 | 4.695 | 6.081 |
| PoinTr | 1.50 | 8.853 | 9.852 | 1.61 | 7.821 | 8.814 |
| Periodic spline | 1.77 | 198.399* | 202.430* | 1.60 | 8.668 | 9.836 |
| ID-1024 (no completion) | 3.28 | 2.296 | — | 1.97 | 1.757 | — |
| ID-8192 (no completion) | 2.56 | 1.503 | — | 1.12 | 1.197 | — |

*Validation mean dominated by extreme failures (median 4.723%/5.969%).

Key findings (see the manuscript for full statements):

- **Geometric ranking does not represent volume ranking**, and the volume ranking itself depends on the reference definition (e.g., IDW and periodic spline beat or match PCN on CD1 while their V_field errors are 2.0x and 3.8x those of PCN on the test split).
- **Volume comparison is sensitive to integration coverage.** On full support, the masked configuration (1.587% V_field, 402.2 integrable slices) edges ID-1024 (1.757%, 157.1 slices); restricted to the three-way common support (155.6 slices) the ordering reverses: **+1.115 percentage points [+0.882, +1.371], 50/50 cavities same-direction**. This direction is unchanged across nine grid–operator combinations (+0.899 to +1.960 pp) and across three runs on the validation split (+0.952/+1.282/+1.345 pp).
- **Benchmark stability is a property of the benchmark.** Four of five arbitrary-initialization repeats degrade both configurations (masked V_field 6.745% in the only converged repeat vs 1.459% main run); data-order changes under fixed initialization produce validation V_field ranges of the same order as the cavity-level interval width; neither is captured by cavity-level bootstrap intervals alone.

## Repository layout

```
SaltCavern-PCC/
├── configs/            # training configurations (BM / E2 / PoinTr arms, fixed-init repeats)
├── code/
│   ├── datagen/        # Sansmic batch build, statistical checks, missingness synthesizer, dataset assembly
│   ├── models/         # training entry points and own glue (dataset/losses/validator) on top of AdaPoinTr
│   ├── baselines/      # rule-based baselines (Cyl-NN / spline / IDW), PCN, identity no-completion baselines
│   ├── evaluation/     # CD1, volume pipeline (single/dual reference, N_slice, common support), surface/cell metrics
│   ├── stats/          # paired bootstrap / Wilcoxon, main endpoint, stability analyses
│   └── figures/        # scripts that reproduce the paper figures
├── third_party/        # PoinTr compatibility patch and notes (the PoinTr library itself is NOT vendored)
├── docs/               # dataset card, results tables, reproducibility map
├── data/            # split list and dataset card (large binaries are Release assets)
├── predictions/     # normalised prediction tree + MANIFEST + SHA256SUMS (binaries not in git)
├── results/         # per-sample metrics behind every table
└── checkpoints/     # checkpoint pointers (weights are Release assets)
```

## Data and model availability

- **Code** is hosted in this repository (MIT license).
- **Dataset (~200 MB)**, **checkpoints** and **exported predictions** are distributed as the assets of the GitHub Release tagged `data-v1.0` and are not tracked in git. Every archive ships with SHA-256 fingerprints; verify with `code/datagen/verify_assets.py`. Split lists and per-sample metrics are small and stay in the repository (`data/splits/`, `results/`).
- The benchmark is released in the recorded coordinate frame (radial coordinates converted to metres, vertical coordinates retaining foot values), together with the affine map z_m = 0.3048 z along the vertical. Volume-class conclusions are invariant to this map; distance-class conclusions hold only in the post-map frame (see manuscript §3.1).
- The test split was used once, under a pre-registered embargo protocol, for the main-endpoint confirmation; it is included for full reproducibility together with the split lists.

## Getting started

**Zero-download check (numpy only).** The bundled cavity in `examples/` reproduces the frozen
closure-and-integration pipeline and matches four anchored entries of
`results/nine_config_per_sample.csv` exactly:

```bash
python examples/demo_dual_reference.py --root examples
```

Expected output: `examples/expected_output.txt`; conventions and the anchor table: `examples/README.md`.

**Where the scripts read data from.** Every analysis script resolves its inputs through the
environment variable `SCC_ROOT` (default `/data`), plus two optional overrides:

```bash
export SCC_ROOT=/path/to/benchmark_root   # must contain dataset/ and validate/
export SCC_ARCHIVE=/path/to/archives      # optional, figure source archives
export SCC_SUPP=supplementary             # optional, supplementary directory name
```

Expected tree under `SCC_ROOT`:

```
$SCC_ROOT/
├── dataset/pc_v2/
│   ├── samples.csv                 # identical to data/splits/samples.csv in this repo
│   ├── val/{partial,complete,gt_field}/
│   └── test/{partial,complete,gt_field}/
├── validate/
│   ├── volume/{gt,pred_adapointr_masked,pred_adapointr_full,...}/
│   └── test_predictions/{recovered/{gt,pred_*},metrics}/
└── supplementary/                  # exported predictions and intermediate artefacts
```

The data archive on the Releases page unpacks into exactly this tree.

**Full pipeline:**

```bash
# 1. environment
pip install -r requirements.txt   # torch / numpy / pandas / scipy / matplotlib / PyYAML / easydict

# 2. AdaPoinTr backbone (NOT included in this repository):
#    clone https://github.com/yuxumin/PoinTr (Apache-2.0) and follow
#    third_party/README.md; the adaptation is already merged into
#    code/models/train_pilot.py (two unpack points in the training entry)

# 3. data: extract the data archive into data/pc_v2/ (see docs/DATASET.md)

# 4. train (one full run ≈ 1.5 GPU-hours on an RTX 4090)
python code/models/train_pilot.py --config configs/pc_v2_pilot_bm.yaml

# 5. evaluate: export predictions, then run the metric pipelines
python code/evaluation/export_adapointr_test.py            # deep-model export
python code/evaluation/baseline_recon.py           # rule-based baselines
python code/evaluation/val_volume.py      # volume / N_slice (val)
python code/evaluation/val_common_support.py      # common support (val)
python code/stats/common_support_val_main.py     # main endpoint (test)
```

Most analysis scripts keep the original instance-side path root `/data`; adjust the root variables at the top of each script when running elsewhere. Script-to-paper mapping: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository self-check

python tools/check_repository.py verifies that every arm directory, configuration and script
referenced by the code and the documentation actually exists, and that all Python files parse.
It exits non-zero on any mismatch.

## Reproducibility notes

- Comparative conclusions use paired same-cavity designs; the main endpoint uses cavity-level clustered bootstrap (10000 resamples, seed 20261103) with exact Wilcoxon sign-rank and sign tests.
- With n = 50 cavities the minimum attainable two-sided sign-test p-value is ≈ 1.8e-15; it certifies sign consistency only, not effect size.
- All geometric computation on large-coordinate point clouds is performed in float64 / normalized coordinates to avoid catastrophic cancellation; all reported geometric metrics are in the physical-metric frame (z × 0.3048).
- Run-to-run stability was characterized along three dimensions: data order (three runs), initialization (five repeats), and integration configuration (nine grid–operator combinations); cavity-level intervals condition only on cavity sampling (manuscript §3.8, §4.5).

## Naming convention

File, checkpoint and configuration names keep the codenames used in the released artifacts, so that checkpoints, logs and fingerprints cross-reference without renaming. Paper-formal names:

| Codename | Paper name |
|---|---|
| `BM` / `pc_v2_pilot_bm*.yaml` | AdaPoinTr, masked (blind-zone) supervision |
| `E2` | AdaPoinTr, full-region supervision |
| `ms_*` | fixed-initialization data-order repeats |
| `B1` / `B2` / `B3` (in baseline scripts) | Cyl-NN / periodic spline / IDW |
| `pc_v2` | the benchmark dataset |
| `s21` in file names | initialization seed 20260921; data/order seeds 20260921 / 20261001 |

## Citation

```bibtex
@misc{saltcavern_pcc_2026,
  title  = {A paired simulation benchmark for sonar salt-cavern point cloud completion: structured missingness, decoupled geometry--volume evaluation, and training-stability checks},
  author = {SaltCavern-PCC authors},
  year   = {2026},
  note   = {Manuscript in preparation; data and code DOI to be registered after acceptance}
}
```

## License

Code: MIT (see [LICENSE](LICENSE)). Dataset: CC BY 4.0. Third-party components (AdaPoinTr/PoinTr, PCN reimplementation, Sansmic) retain their original licenses; see [third_party/README.md](third_party/README.md).

## Disclaimer

All conclusions are drawn on the simulation benchmark and the reported runs. Statistical perturbations mitigate but do not eliminate the axisymmetric nature of the simulated truth; the missingness parameters are not calibrated against field sonar statistics. Validation on real sonar surveys of operating caverns is the primary follow-up direction.
