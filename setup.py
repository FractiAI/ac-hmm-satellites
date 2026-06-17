from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "achmm_trellis",
        ["src/trellis.cpp", "src/bindings.cpp"],
        include_dirs=["src"],
        cxx_std=17,
    ),
]

setup(
    name="achmm-satellites",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
