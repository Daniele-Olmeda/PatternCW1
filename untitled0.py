# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 18:07:21 2018

@author: danie
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,1,2,3])
y = np.array([0.650, 0.660, 0.675, 0.685])
my_xticks = ['a', 'b', 'c', 'd']
plt.xticks(x, my_xticks)
plt.yticks(np.arange(y.min(), y.max(), 0.005))
plt.plot(x, y)
plt.grid(axis='y', linestyle='-')
plt.show()