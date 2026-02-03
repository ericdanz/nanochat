"""
Setup script for building the fused look-around flash attention CUDA extension.

Build with:
    cd nanochat/cuda_kernels
    pip install -e .

Or build in-place:
    python setup.py build_ext --inplace
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this setup.py
this_dir = os.path.dirname(os.path.abspath(__file__))
csrc_dir = os.path.join(this_dir, 'csrc')

# CUDA compute capabilities to target
# Targeting H200 (sm_90) and RTX 5090 (sm_120) for bf16 WMMA tensor core support
CUDA_ARCH_LIST = [
    '-gencode=arch=compute_90,code=sm_90',   # Hopper (H200) - bf16 WMMA
    '-gencode=arch=compute_120,code=sm_120', # Blackwell (RTX 5090) - bf16 WMMA
]

# Compiler flags
CXX_FLAGS = [
    '-O3',
    '-std=c++17',
]

NVCC_FLAGS = [
    '-O3',
    '--use_fast_math',
    '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '-lineinfo',  # Enable line info for profiling
] + CUDA_ARCH_LIST

setup(
    name='fused_look_around_flash_cuda',
    version='0.1.0',
    description='Fused look-around flash attention CUDA kernel',
    author='NC2 Project',
    ext_modules=[
        CUDAExtension(
            name='fused_look_around_flash_cuda',
            sources=[
                os.path.join(csrc_dir, 'fused_look_around_flash.cu'),
                os.path.join(csrc_dir, 'bindings.cpp'),
            ],
            include_dirs=[csrc_dir],
            extra_compile_args={
                'cxx': CXX_FLAGS,
                'nvcc': NVCC_FLAGS,
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0',
    ],
)
