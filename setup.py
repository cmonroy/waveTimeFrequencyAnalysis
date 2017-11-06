#!/usr/bin/env python
from distutils.core import setup

package_dir= { "dropy" : '.' ,
              "dropy.Reader" : r"./Reader" ,
              "dropy.pyplotTools" : r"./pyplotTools" ,
              "dropy.TimeDomain" : r"./TimeDomain" ,
             }

packages = package_dir.keys()

setup(name='dropy',
      version='1.0',
      description='DR open library, related to waves and sea-keeping',
      author='DR',
      author_email='',
      url= r'https://github.com/BV-DR/waveTimeFrequencyAnalysis',
      packages = packages ,
      package_dir = package_dir ,
      install_requires = [ "pandas>=0.20" , "numpy" , "scipy" , "matplotlib>=1.5" ] ,
      classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
     )