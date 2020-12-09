import os
import sys
import numpy as np

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

iculibs = ['icuuc'] if os.name == 'nt' else ['icuuc', 'icudata']

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Manufacturing
Intended Audience :: Information Technology
Intended Audience :: Developers
License :: OSI Approved :: The Unlicense (Unlicense)
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3 :: Only
Topic :: Software Development
Topic :: Database
Topic :: Software Development :: Libraries
Operating System :: Microsoft :: Windows :: Windows 10
Operating System :: POSIX
Operating System :: OS Independent
Operating System :: Unix
Operating System :: MacOS :: MacOS X
"""

setup(
    name='fastdbf',
    version='0.1.5',
    description='Read DBF Files with C and Python, fast.',
    long_description='Reads DBF files into a pandas DataFrame using a C extension when available, or a fallback pure Python parser.',
    author='Raphael Campestrini',
    author_email='rca@seca.at',
    packages = ['fastdbf'],
    include_package_data=True,
    zip_safe=True,
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms=["Windows", "Linux", "Mac OS-X", "Unix"],
    python_requires='>=3.6',
    url="https://pypi.python.org/pypi/fastdbf",
    download_url="https://pypi.python.org/pypi/fastdbf",
    install_requires=[
        'pandas',
        'numpy'
    ],
    ext_modules=[
        Extension(
            '_fastdbf',
            sources=['_fastdbfmodule.c'],
            include_dirs=[np.get_include()],
            libraries=iculibs,
            optional=True
        )
    ]
)
