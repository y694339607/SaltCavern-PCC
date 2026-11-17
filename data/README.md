# data/

Placeholder for the released dataset; the split list lives in `data/splits/samples.csv`.

The full benchmark (500 cavities x 3 coverage tiers = 1500 paired samples) is the
`dataset_pc_v2.zip` asset of the `data-v1.0` release: 429,379,988 bytes, SHA-256
`83086a5b2e9896eb1447e632a1872b07e1f1846b80ad7e2defe5f98b091ba7e9`. Unpack it so that
`$SCC_ROOT/dataset/pc_v2/{val,test}` exists; the tree is documented in `docs/DATASET.md`.

数据集发布位。完整数据（500 腔 × 3 档 = 1500 对）为 `data-v1.0` 发布中的 `dataset_pc_v2.zip`：
429,379,988 字节，SHA-256 `83086a5b…a7e9`。解压后使 `$SCC_ROOT/dataset/pc_v2/{val,test}` 存在，
目录结构见 `docs/DATASET.md`。划分清单在仓库内：`data/splits/samples.csv`。
