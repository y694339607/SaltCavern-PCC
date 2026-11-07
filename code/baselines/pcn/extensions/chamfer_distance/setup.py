from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 根据您的显卡和 CUDA 版本调整
# 4060Ti 对应 compute_89 (Ada Lovelace)
# 如果 CUDA 11.x，可能需要 compute_86 (Ampere)

setup(
    name='chamfer_3D',
    ext_modules=[
        CUDAExtension(
            'chamfer_3D',
            [
                'chamfer_cuda.cpp',
                'chamfer3D.cu',
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    #'-gencode', 'arch=compute_86,code=sm_86',  # 3060/4060 系列
                    '-gencode', 'arch=compute_89,code=sm_89',  # 4060Ti 等 Ada
                ],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)