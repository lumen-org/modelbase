from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

# Include numpy
numpy_include = numpy.get_include()
os.environ["CFLAGS"] = "-I" + numpy_include
include_path = [numpy_include]

setup(
    name="graphical model sampling",
    ext_modules=cythonize("gen_samples.pyx"),
    include_dirs=include_path
)
