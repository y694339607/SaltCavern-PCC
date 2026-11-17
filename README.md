# SaltCavern-PCC

**Paired benchmark and decoupled geometry–volume evaluation protocol for sonar-based salt-cavern point-cloud completion**

Code accompanying the manuscript *Paired benchmark and decoupled geometry–volume evaluation protocol for sonar-based salt-cavern point-cloud completion* (manuscript in preparation).

---

## About

Salt-cavern sonar surveys sample discretely in the vertical direction; the resulting missingness is layered, cylindrical and geometry-dependent rather than random. Storage capacity is a scalar obtained by integrating a closed surface, so geometric metrics alone do not answer whether the volume is usable.

This repository provides:

1. **A paired simulation benchmark.** 500 cavities generated with the open-source leaching simulator [Sansmic](https://github.com/sandialabs/sansmic) (400 with statistical perturbations, 100 axisymmetric controls), observed under three coverage tiers (L0/L1/L2, nominal retention 70%/50%/30%) synthesized from four sonar observation mechanisms plus an angular-resolution representation limit, giving **1500 paired samples** (incomplete 8192 points stored / model input 1024 points / complete and deep-model output 16384 points). Splits are cavity-isolated (train/val/test = 400/50/50 cavities).

2. **An evaluation protocol that separates geometric error from volume error.** Two reference volumes (an independent simulated radius field V_field, and V_ref obtained by running the complete cloud through the same closure-and-integration pipeline), integrable-slice counting (N_slice), three-way common support (N_common), and input-matched no-completion baselines (ID-1024).

3. **Robustness analyses.** Nine grid–operator combinations, three data-order runs, an initialization stratum, density stratification and three negative controls (`code/stats/`).

4. **Reproducible delivery.** Fixed split lists (`data/splits/samples.csv`), the per-sample metrics behind every table (`results/`), and the integration and statistics code in this repository.

## Main results (physical metric units)

Overall table (one protocol run, val/test):

| Method | Val CD1 (m) | Val V_field (%) | Val V_ref (%) | Test CD1 (m) | Test V_field (%) | Test V_ref (%) |
|---|---|---|---|---|---|---|
| Cylindrical nearest-neighbor (Cyl-NN) | 0.90 | 1.660 | 0.754 | 0.92 | 1.643 | 0.635 |
| AdaPoinTr, masked supervision | 0.90 | 1.459 | 1.628 | 0.97 | 1.587 | 1.889 |
| AdaPoinTr, full-region supervision | 0.95 | 1.468 | 2.123 | 1.00 | 1.765 | 2.322 |
| PCN | 1.52 | 2.538 | 3.210 | 1.62 | 2.292 | 2.844 |
| IDW | 1.19 | 4.260 | 5.618 | 1.27 | 4.695 | 6.081 |
| PoinTr | 1.50 | 8.853 | 9.852 | 1.61 | 7.821 | 8.814 |
| Periodic spline | 1.77 | 198.399* | 202.430* | 1.60 | 8.668 | 9.836 |
| ID-1024 (no completion) | 1.87 | 1.843 | — | 1.97 | 1.757 | — |

*Validation mean dominated by extreme failures (median 4.723%/5.969%).

## Repository layout

```
SaltCavern-PCC/
├── configs/            # training configurations (BM / E2 / PoinTr configurations, fixed-init repeats)
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

- **Code** is hosted in this repository (MIT license, see `LICENSES.md` for the data and
  third-party terms).
- **Dataset (~410 MB)**, **checkpoints** and **exported predictions** are distributed as the assets of the GitHub Release tagged `data-v1.0` and are not tracked in git. Every archive ships with SHA-256 fingerprints; verify with `code/datagen/verify_assets.py`. Split lists and per-sample metrics are small and stay in the repository (`data/splits/`, `results/`).
- The benchmark is released in the recorded coordinate frame (radial coordinates converted to metres, vertical coordinates retaining foot values), together with the affine map z_m = 0.3048 z along the vertical. Volume-class conclusions are invariant to this map; distance-class conclusions hold only in the post-map frame (see manuscript §3.1).
- The test split is included for full reproducibility together with the split lists.

## Getting started

**Zero-download check (numpy only).** The bundled cavity in `examples/` reproduces the frozen
closure-and-integration pipeline:

```bash
python examples/demo_dual_reference.py --root examples
```

Expected output: `examples/expected_output.txt`.

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
python code/models/train_pilot.py --config configs/adapointr_masked.yaml

# 5. evaluate: export predictions, then run the metric pipelines
python code/evaluation/export_adapointr_test.py            # deep-model export
python code/evaluation/baseline_recon.py           # rule-based baselines
python code/evaluation/val_volume.py      # volume / N_slice (val)
python code/evaluation/val_common_support.py      # common support (val)
python code/stats/common_support_val_main.py     # main endpoint (test)
```

Most analysis scripts keep the original instance-side path root `/data`; adjust the root variables at the top of each script when running elsewhere. Script-to-paper mapping: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository self-check

python tools/check_repository.py verifies that every prediction directory, configuration and script
referenced by the code and the documentation actually exists, and that all Python files parse.
It exits non-zero on any mismatch.

## Reproducibility notes

- Comparative endpoints use paired same-cavity designs and cavity-level clustered bootstrap (10000 resamples, seed 20261103) with exact sign tests.
- Geometric computation uses float64 / normalized coordinates; all reported metrics are in the physical-metric frame (z × 0.3048).

## Naming convention

Per-sample CSV columns and analysis variables keep the short method tags used in the released
per-sample files; paper-formal names:

| Tag (CSV columns / analysis code) | Paper name |
|---|---|
| `BM` | AdaPoinTr, masked (blind-zone) supervision |
| `E2` | AdaPoinTr, full-region supervision |
| `B1` / `B2` / `B3` | Cyl-NN / periodic spline / IDW |
| `pc_v2` | the benchmark dataset |

## Citation

```bibtex
@misc{saltcavern_pcc_2026,
  title  = {Paired benchmark and decoupled geometry--volume evaluation protocol for sonar-based salt-cavern point-cloud completion},
  author = {SaltCavern-PCC authors},
  year   = {2026},
  note   = {Manuscript in preparation; data and code DOI to be registered after acceptance}
}
```

## License

Code: MIT (see [LICENSE](LICENSE)). Dataset: CC BY 4.0. Third-party components (AdaPoinTr/PoinTr, PCN reimplementation, Sansmic) retain their original licenses; see [third_party/README.md](third_party/README.md).

## Disclaimer

All conclusions are limited to the simulated benchmark and the reported runs. Statistical perturbations mitigate but do not eliminate the axisymmetric nature of the simulated truth; the missingness parameters are not calibrated against field sonar statistics.
