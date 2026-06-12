from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy


extensions = [
  Extension(
    name="core",
    sources=["core.pyx"],
    include_dirs=[numpy.get_include()],
  )
]

setup(
  name="monotonic_align",
  ext_modules=cythonize(extensions),
)
