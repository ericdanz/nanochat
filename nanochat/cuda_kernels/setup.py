"""
Setup script for building the fused look-around flash attention CUDA extension.

Build with:
    cd nanochat/cuda_kernels
    pip install -e .

Or build in-place:
    python setup.py build_ext --inplace

Requirements:
    - CUDA 12.x for SM90 (Hopper) support
    - CUDA 13.0+ for SM120 (Blackwell) support with tcgen05/TMEM
"""

import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this setup.py
this_dir = os.path.dirname(os.path.abspath(__file__))
csrc_dir = os.path.join(this_dir, 'csrc')
cutlass_dir = os.path.join(this_dir, 'cutlass')

# Include directories
include_dirs = [
    csrc_dir,
    os.path.join(cutlass_dir, 'include'),
    os.path.join(cutlass_dir, 'tools/util/include'),
]


def get_cuda_version():
    """Get CUDA version from nvcc."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        output = result.stdout
        # Parse "release X.Y" from nvcc output
        import re
        match = re.search(r'release (\d+)\.(\d+)', output)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    return None, None


cuda_major, cuda_minor = get_cuda_version()
print(f"Detected CUDA version: {cuda_major}.{cuda_minor}")

# CUDA compute capabilities to target
# Targeting H200 (sm_90) and RTX 5090 (sm_120) for bf16 WMMA/WGMMA tensor core support
CUDA_ARCH_LIST = [
    '-gencode=arch=compute_90,code=sm_90',   # Hopper (H200) - WGMMA, TMA
    '-gencode=arch=compute_90a,code=sm_90a', # Hopper with async features
]

# Add SM120 (Blackwell) support if CUDA >= 13.0
if cuda_major is not None and cuda_major >= 13:
    # Only use sm_120a - sm_120 doesn't support F8 MMA features
    # The RTX 5090 reports as sm_120 but should run sm_120a code via JIT
    CUDA_ARCH_LIST.append('-gencode=arch=compute_120a,code=sm_120a') # Blackwell with F8 MMA
    # Also embed PTX for forward compatibility
    CUDA_ARCH_LIST.append('-gencode=arch=compute_120a,code=compute_120a')
    print("SM120a (Blackwell) support enabled with CUDA 13.0+")
else:
    print(f"SM120 (Blackwell) support disabled (requires CUDA 13.0+, found {cuda_major}.{cuda_minor})")

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
    '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
    '-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED',
]

# Add SM120-specific defines if CUDA 13.0+
if cuda_major is not None and cuda_major >= 13:
    NVCC_FLAGS.extend([
        '-DCUTE_SM120_EXTENDED_MMA_SHAPES_ENABLED',
        '-DENABLE_SM120=1',
        # Enable F8/F6/F4 MMA atoms for SM120 (requires CUDA 12.8+)
        # These defines enable CUTLASS_ARCH_MMA_SM120_ENABLED and CUTE_ARCH_F8F6F4_MMA_ENABLED
        '-DCUTLASS_ARCH_MMA_SM120_SUPPORTED=1',
    ])
    # Also add to CXX flags for bindings.cpp
    CXX_FLAGS.append('-DENABLE_SM120=1')

NVCC_FLAGS.extend(CUDA_ARCH_LIST)

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
            include_dirs=include_dirs,
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
