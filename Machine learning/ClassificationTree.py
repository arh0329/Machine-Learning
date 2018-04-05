#Aaron Henningfeld
#P04

import numpy as np 
import pandas as pd

class ClassificationTree:

    def __init__(self, X, y, max_depth=2, depth=0, min_leaf_size=2, classes=None):
        
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_obs = len(X) #find num of observations
        self.classes = np.unique(self.y)# find the classes
        self.depth = depth
        