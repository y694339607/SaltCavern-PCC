# setup_ext.py — 编译 chamfer 扩展（实例本地 nvcc，无需外网）
import torch
from pathlib import Path
mod = torch.utils.cpp_extension.load(
    name='chamfer_3D',
    sources=[str(Path(__file__).parent/'extensions/chamfer_distance/chamfer_cuda.cpp'),
             str(Path(__file__).parent/'extensions/chamfer_distance/chamfer3D.cu')],
    verbose=True)
print('ext ok:', mod)
