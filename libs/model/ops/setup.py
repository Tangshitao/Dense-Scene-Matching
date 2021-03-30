#!/usr/bin/env python

import os

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_file = os.path.dirname(__file__)

setup(
    name="correlation_package",
    ext_modules=[
        CUDAExtension(
            "correlation_cuda",
            [
                "correlation/src/corr_cuda.cpp",
                "correlation/src/corr.cpp",
                "correlation/src/corr_cuda_kernel.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        ),
        CUDAExtension(
            "correlation_proj",
            ["correlation/src/corr_proj.cpp", "correlation/src/corr_proj_kernel.cu"],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        ),
        CUDAExtension(
            "nms",
            ["nms/src/nms.cpp", "nms/src/nms_kernel.cu"],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
