
from setuptools import setup

setup(
    name="SwiftLLM",
    version="0.0.1",
    author="Shengyu Liu",
    description="A tiny yet powerful LLM inference system tailored for researching purpose",
    packages=[
        "swiftllm",
        "swiftllm.server",
        "swiftllm.worker",
        "swiftllm.worker.kernels",
        "swiftllm.worker.layers",
    ],
    zip_safe=False,
)
