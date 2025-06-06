from setuptools import setup
from Cython.Build import cythonize
import numpy

'''
Use this code compile extract_sum with Cython

Compile with: python3 compile.py build_ext --inplace
'''

# For multiprocessing in Python
setup(
    ext_modules=cythonize("extract_sum_mp.pyx"),
    include_dirs=[numpy.get_include()]
)

# For single process in Python
# setup(
#     ext_modules=cythonize("extract_sum.pyx"),
#     include_dirs=[numpy.get_include()]
# )