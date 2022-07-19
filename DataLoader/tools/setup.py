# Copyright (c) 2022 Dai HBG


from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("calculate_tools.pyx", annotate=True),
)
