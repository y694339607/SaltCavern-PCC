# Third-party compatibility

## pointr_compat.patch

Compatibility adaptation for the PoinTr (AdaPoinTr) backbone — **two unpack
points** in the training entry point, allowing both 4-tuple
`(coarse, _, _, fine)` and 2-tuple `(coarse, fine)` model returns:

```bash
patch -p1 < third_party/pointr_compat.patch   # applies the two unpack points to your own copy
```

The adaptation is already incorporated in `code/models/train_pilot.py`
(the version used for the PoinTr arm and the re-runs); a full recursive diff
against the official PoinTr tree showed no library-code differences, i.e. no
patch to the PoinTr library itself is required.

License: PoinTr (PointTr) is Apache-2.0, (c) its authors; see
https://github.com/yuxumin/PoinTr
