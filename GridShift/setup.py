from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext


setup(
    name='GridShiftPP',
    version='1.0',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("GridShiftPP",
                 sources=["gridshiftpp.pyx"],
                 language="c++",
                 include_dirs=[numpy.get_include()])],
    author='Abhishek Kumar',
    author_email='abhishek.kumar.eee13@itbhu.ac.in')
