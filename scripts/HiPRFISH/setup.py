from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize('neighbor.pyx', annotate = True),
    include_dirs = [np.get_include()]
)
