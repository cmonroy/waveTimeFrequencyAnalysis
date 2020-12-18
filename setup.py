#!/usr/bin/env python
from setuptools import setup

package_dir= { "droppy" : 'droppy' ,
               "droppy.Reader" : r"./droppy/Reader" ,
               "droppy.pyplotTools" : r"./droppy/pyplotTools" ,
               "droppy.TimeDomain" : r"./droppy/TimeDomain" ,
               "droppy.numerics" : r"./droppy/numerics" ,
               "droppy.Form" : r"./droppy/Form" ,
               "droppy.interpolate" : r"./droppy/interpolate" ,
             }

packages = package_dir.keys()

setup(name='droppy',
      version='1.0.1',
      description='DR open library, related to waves and sea-keeping',
      author='DR',
      author_email='',
      url= r'https://github.com/BV-DR/droppy',
      packages = packages ,
      package_dir = package_dir ,
      install_requires = [ "pandas>=0.20" , "numpy" , "scipy" , "matplotlib>=1.5" ] ,
      classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
     )
