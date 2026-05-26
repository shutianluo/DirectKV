import os
from setuptools import setup
from torch.utils import cpp_extension

try:
    import pybind11
    _pybind11_inc = [pybind11.get_include()]
except ImportError:
    _pybind11_inc = []

__version__ = "0.0.1"

ext_modules = [
    cpp_extension.CUDAExtension(
        "swiftllm_c",
        [
            "src/entrypoints.cpp",
			"src/block_swapping.cpp",
            "src/small_kernels.cu",
            # "src/attention.cu",
            "src/linear.cu",
        ],
        include_dirs=_pybind11_inc,
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math']
        }
    ),
]

setup(
    name="swiftllm_c",
    version=__version__,
    author="Shengyu Liu",
    author_email="shengyu.liu@stu.pku.edu.cn",
    url="",
    description="Some C++/CUDA sources for SwiftLLM.",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    zip_safe=False,
    python_requires=">=3.9",
)
