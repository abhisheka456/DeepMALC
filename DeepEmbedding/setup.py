import os
import numpy
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
# from numpy.distutils.core import Extension
from Cython.Build import cythonize


def readme():
    with open('README.md') as readme_file:
        return readme_file.read()


# os.environ["CC"] = "clang"
extensions = [
    Extension(
        "DE._utils_tsne",
        ["DE/_utils_tsne.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]


configuration = {
    'name' : 'DE',
    'version': '1.0.0',
    'description' : 'Deep Embedding for High-Dimensional Data',
    'long_description' : readme(),
    'long_description_content_type' : 'text/markdown',
    'classifiers' : [
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        "License :: OSI Approved :: MIT License",
    ],
    'keywords' : 'DeepMALC',
    'url' : '.....',
    'author' : '....',
    'author_email' : '.....',
    'license' : 'LICENSE',
    'packages' : find_packages(),
    'setup_requires' : ["cython", "numpy"],
    'install_requires' : ['scikit-learn >= 0.16',
                          'numba >= 0.34',
                          'torch >= 1.0',
                          'tqdm',
                          'ipywidgets'],
    'ext_modules' : cythonize(extensions),
    'include_dirs' : [numpy.get_include()],
    }

setup(**configuration)
