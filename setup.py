# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:10:26 2020

@author: Amin
"""

#!/usr/bin/env python

from distutils.core import setup

setup(name='ccm',
      version='1.0',
      description='Methods for simulating neural data and computing causality and prediction of time series data based on convergent cross mapping.',
      author='Amin Nejatbakhsh',
      author_email='mn2822@columbia.net',
      url='https://github.com/amin-nejat/',
      install_requires=['networkx', 'numpy', 'scipy', 'matplotlib', 'scikit_learn'],
      packages=['Simulation', 'DelayEmbedding','Causality'],
     )
