 # -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:24:16 2020
"""

import numpy as np
from DelayEmbedding import DelayEmbedding as DE 
x=np.random.rand(2,100)
F_matrix= DE.connectivity(x)

             
