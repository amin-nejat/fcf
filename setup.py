# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:10:26 2020

@author: Amin
"""

#!/usr/bin/env python

from distutils.core import setup

setup(name='ccm',
      version='1.0',
      description='Methods for Simulating Spiking and Rate Networks and Computing Causality and Prediction of Time Series Data Based on Convergent Cross Mapping.',
      author='Amin Nejatbakhsh',
      author_email='mn2822@columbia.net',
      url='https://github.com/amin-nejat/',
      packages=['simulator', 'delay_embedding','causality'],
     )
