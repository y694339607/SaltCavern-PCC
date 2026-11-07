# 第三方兼容性

## pointr_compat.patch

PoinTr（AdaPoinTr）骨干的兼容性适配——训练入口中的**两处解包点**，同时兼容 4 元组 `(coarse, _, _, fine)` 与 2 元组 `(coarse, fine)` 两种模型返回形式：

```bash
patch -p1 < third_party/pointr_compat.patch   # 为自己的副本补上两处解包
```

该适配已并入 `code/models/train_pilot.py`；对官方 PoinTr 树的全量递归 diff 显示库代码无任何差异，即无需对 PoinTr 库本体打补丁。

许可证：PoinTr（PointTr）为 Apache-2.0，版权归其作者所有；见 https://github.com/yuxumin/PoinTr
