# code/stats/

Paired statistics, the main endpoint, and the stability analyses.

## Common-support endpoint scripts

Five scripts compute the same endpoint and differ only in which split and which
prediction set they read. They are kept as-is because they produced the numbers
reported in the manuscript; a single parameterised entry point is planned.

| Script | Split | Prediction set |
|---|---|---|
| `common_support_test.py` | test | main run, masked arm |
| `common_support_test_altlayout.py` | test | same, with the v2 path layout |
| `common_support_val_main.py` | val | main run, masked arm |
| `common_support_val_order1.py` | val | data-order repeat 1 |
| `common_support_val_order2.py` | val | data-order repeat 2 |

All five share the frozen closure-and-integration pipeline (`500 x 72` grid,
in-cell maximum radius, >= 3 valid azimuth cells, >= 10 integrable slices,
trapezoid over slice centres) and the ID-1024 draw
`Philox(SeedSequence(20260921, spawn_key=[cavity, level, 2])).permutation(8192)[:1024]`.

## Other scripts

- `repeats_eval.py`, `repeats_stats.py` — masked vs full-region supervision, fixed-initialization repeats (Table 7).
- `init_eval.py`, `init_stats.py`, `init_fingerprint.py` — initialization stratum (Section 4.5).
- `negative_control_eval.py` — negative controls and initialization summary (Section 4.5).
- `supervision_arms_cd1.py` — paired CD1 differences of the two supervision arms (Section 4.3).
