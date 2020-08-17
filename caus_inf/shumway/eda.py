"""
Scripts to do the EDA of chapter 2 in Shumway and Stoffer
"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression

def detrend(Y):
    """
    Returns detrended version of Y
    """
    
    X = np.array( range( len( Y) ) ).reshape(-1, 1)
    reg = LinearRegression().fit(X, Y)
    
    trend = reg.predict(X)
    detY = Y - trend

    return detY

