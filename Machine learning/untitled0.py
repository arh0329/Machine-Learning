# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:16:30 2018

@author: arhen
"""

import matplotlib.pyplot as plt
x = range(0,5)
y1 = [0.1, 0.2, 0.5, 0.3, 0.3] 
y2 = [0.3, 0.6, 0.8, 0.7, 0.5] 
plt.close()
plt.plot(x, y1)
plt.plot(x, y2)
plt.ylim([0,1])
plt.show()