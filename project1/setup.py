from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "linear_regression",
        ["linear_regression.cpp"],
    ),
]

setup(
    name="linear_regression",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)