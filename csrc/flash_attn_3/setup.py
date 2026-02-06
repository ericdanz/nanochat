# Simplified FA3 build for nanochat: H100 (sm90a), bf16, hdim128 only.
# Adapted from upstream flash-attention/hopper/setup.py

import os
import re
import ast
from pathlib import Path

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "flash_attn_3"

# ---------- monkey-patch ninja file to dispatch sm90a arch ----------
from torch.utils.cpp_extension import (
    SUBPROCESS_DECODE_ARGS,
    IS_WINDOWS,
    get_cxx_compiler,
    _join_cuda_home,
    _is_cuda_file,
    _maybe_write,
)

def _write_ninja_file(path, cflags, post_cflags, cuda_cflags, cuda_post_cflags,
                      cuda_dlink_post_cflags, sources, objects, ldflags,
                      library_target, with_cuda, **kwargs) -> None:
    def sanitize_flags(flags):
        return [flag.strip() for flag in flags] if flags else []

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    assert len(sources) == len(objects) and len(sources) > 0

    compiler = get_cxx_compiler()
    config = ['ninja_required_version = 1.3', f'cxx = {compiler}']
    if with_cuda or cuda_dlink_post_cflags:
        nvcc = _join_cuda_home('bin', 'nvcc')
        nvcc_from_env = os.getenv("PYTORCH_NVCC", nvcc)
        config += [f'nvcc_from_env = {nvcc_from_env}', f'nvcc = {nvcc}']

    flags = [f'cflags = {" ".join(cflags)}', f'post_cflags = {" ".join(post_cflags)}']
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
    flags += [f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}',
              f'ldflags = {" ".join(ldflags)}']

    sources = [os.path.abspath(file) for file in sources]

    compile_rule = ['rule compile',
                    '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags',
                    '  depfile = $out.d', '  deps = gcc']

    cuda_compile_rule = []
    if with_cuda:
        nvcc_gendeps = ''
        if torch.version.cuda is not None and os.getenv('TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', '0') != '1':
            nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
        cuda_compile_rule = ['rule cuda_compile',
                             '  depfile = $out.d', '  deps = gcc',
                             f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags']

    # All our .cu files are _sm90.cu, so they all get the default cuda_post_cflags (which has sm_90a)
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        rule = 'cuda_compile' if is_cuda_source else 'compile'
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    devlink_rule, devlink = [], []
    if library_target is not None:
        link_rule = ['rule link', '  command = $cxx $in $ldflags -o $out']
        link = [f'build {library_target}: link {" ".join(objects)}']
        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []

    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks) + "\n"
    _maybe_write(path, content)

torch.utils.cpp_extension._write_ninja_file = _write_ninja_file


# ---------- build config ----------
def create_build_config_file():
    CONFIG = {
        "build_flags": {
            "FLASHATTENTION_DISABLE_BACKWARD": False,
            "FLASHATTENTION_DISABLE_SPLIT": False,
            "FLASHATTENTION_DISABLE_PAGEDKV": True,
            "FLASHATTENTION_DISABLE_APPENDKV": False,
            "FLASHATTENTION_DISABLE_LOCAL": False,
            "FLASHATTENTION_DISABLE_SOFTCAP": True,
            "FLASHATTENTION_DISABLE_PACKGQA": True,
            "FLASHATTENTION_DISABLE_FP16": True,
            "FLASHATTENTION_DISABLE_FP8": True,
            "FLASHATTENTION_DISABLE_VARLEN": False,
            "FLASHATTENTION_DISABLE_CLUSTER": True,
            "FLASHATTENTION_DISABLE_HDIM64": True,
            "FLASHATTENTION_DISABLE_HDIM96": True,
            "FLASHATTENTION_DISABLE_HDIM128": False,
            "FLASHATTENTION_DISABLE_HDIM192": True,
            "FLASHATTENTION_DISABLE_HDIM256": True,
            "FLASHATTENTION_DISABLE_SM8x": True,
            "FLASHATTENTION_ENABLE_VCOLMAJOR": False,
            "FLASH_ATTENTION_DISABLE_HDIMDIFF64": True,
            "FLASH_ATTENTION_DISABLE_HDIMDIFF192": True,
        }
    }
    with open(os.path.join(this_dir, "flash_attn_config.py"), "w") as f:
        f.write("# Auto-generated by flash attention 3 setup.py\n")
        f.write(f"CONFIG = {repr(CONFIG)}\n\n")
        f.write("def show():\n    from pprint import pprint\n    pprint(CONFIG)\n")


# ---------- extension ----------
create_build_config_file()

feature_args = [
    "-DFLASHATTENTION_DISABLE_SM8x",
    "-DFLASHATTENTION_DISABLE_SOFTCAP",
    "-DFLASHATTENTION_DISABLE_PAGEDKV",
    "-DFLASHATTENTION_DISABLE_PACKGQA",
    "-DFLASHATTENTION_DISABLE_FP16",
    "-DFLASHATTENTION_DISABLE_FP8",
    "-DFLASHATTENTION_DISABLE_CLUSTER",
    "-DFLASHATTENTION_DISABLE_HDIM64",
    "-DFLASHATTENTION_DISABLE_HDIM96",
    "-DFLASHATTENTION_DISABLE_HDIM192",
    "-DFLASHATTENTION_DISABLE_HDIM256",
    "-DFLASHATTENTION_DISABLE_HDIMDIFF64",
    "-DFLASHATTENTION_DISABLE_HDIMDIFF192",
]

sources = [
    "flash_api.cpp",
    "instantiations/flash_fwd_hdim128_bf16_sm90.cu",
    "instantiations/flash_fwd_hdim128_bf16_split_sm90.cu",
    "instantiations/flash_bwd_hdim128_bf16_sm90.cu",
    "flash_fwd_combine.cu",
    "flash_prepare_scheduler.cu",
]

cc_flag = ["-gencode", "arch=compute_90a,code=sm_90a"]

nvcc_flags = [
    "-O3", "-std=c++17",
    "--ftemplate-backtrace-limit=0",
    "--use_fast_math",
    "--resource-usage",
    "-lineinfo",
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
]

def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "2"
    return ["--threads", nvcc_threads]

cutlass_dir = Path("/lambda/nfs/speedrun/cutlass")
include_dirs = [Path(this_dir), cutlass_dir / "include"]

ext_modules = [
    CUDAExtension(
        name=f"{PACKAGE_NAME}._C",
        sources=sources,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"] + feature_args,
            "nvcc": nvcc_threads_args() + nvcc_flags + cc_flag + feature_args,
        },
        include_dirs=include_dirs,
    )
]


def get_package_version():
    with open(Path(this_dir) / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    return ast.literal_eval(version_match.group(1))


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=[PACKAGE_NAME],
    package_dir={PACKAGE_NAME: "."},
    py_modules=[],
    description="FlashAttention-3 (nanochat local build, H100 bf16 hdim128)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch", "packaging", "ninja"],
)
