# PCN baseline: provenance and adaptations

## Provenance

Community PyTorch reimplementation of PCN (architecture identical to wentaoyuan/pcn;
self-reported metrics in the original README are of the same magnitude as the paper's).
Reviewed before inclusion: structure, loss form (cd(coarse) + alpha * cd(fine), alpha
ramped by train_step), and optimizer (Adam 1e-4) are consistent with the official
protocol; remaining differences are customary community-port choices.

## Adaptations made in this repository (relative to the reimplementation)

1. `models/pcn.py`: `folding_seed` changed to `register_buffer` (the original hard-coded
   `.cuda()`, which excludes it from `state_dict` and `.to(device)`).
2. Added `dataset_pc.py`: adapter to the shared paired-dataset class (same data chain,
   same splits, same normalization as the AdaPoinTr arms).
3. Added `pcn_loss.py`: unified Chamfer loss — uses the compiled `extensions/chamfer_distance`
   extension when available, falls back to per-sample chunked `cdist`.
4. Added `pcn_train.py`: 250 epochs, seed 20261015; removes tensorboardX/visualization/open3d
   dependencies.
5. Added `pcn_export.py`: exports predictions (denormalized) and an L1-CD table.
6. Training protocol aligned with the AdaPoinTr arms where applicable (250 epochs, same
   seed); optimizer settings (lr, StepLR, batch size, alpha ramp) keep the community
   protocol and are described as such in the manuscript.

## Usage

```bash
cd code/baselines/pcn
python setup_ext.py        # compile the Chamfer distance extension (CUDA required)
python pcn_train.py        # train (~hours depending on GPU)
python pcn_export.py       # export predictions and the CD1 table
```
