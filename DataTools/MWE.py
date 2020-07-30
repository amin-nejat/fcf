# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:40:34 2020

"""



import itertools
nEvents=2
nChannels=3
stimCh=2
s=itertools.product(range(nEvents),(x for x in range(nChannels) if x != stimCh-1))
for x,y in s:
     print(str(x)+' and '+str(y))