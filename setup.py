'''
Created on 12 April 2017
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
# from distutils.core import setup


setup(name='psi',
      version='0.1.1',
      author='Sambit Giri',
      author_email='sambit.giri@astro.su.se',
      package_dir={'': 'src'},
      packages=find_packages(where='src'),
      package_data={'share': ['*']},
      install_requires=[
          'numpy', 'scipy', 'matplotlib',
          'scikit-learn', 'scikit-image',
          'pyDOE', 'getdist',
      ],
)
