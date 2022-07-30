import sys
import glob

from os import path
from setuptools import setup, find_packages
from setuptools.extension import Extension

if sys.version_info < (3,6):
    sys.exit("Sorry, only Python >= 3.6 is supported")
here = path.abspath(path.dirname(__file__))

setup(
    name='OVAL21-generation',
    version='0.0.1',
    description='OVAL21 dataset generation',
    author='OVAL group, University of Oxford',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    install_requires=['sh', 'numpy', 'torch', 'scipy', 'torchvision'],
    extras_require={
        'dev': ['ipython', 'ipdb']
    },
)
