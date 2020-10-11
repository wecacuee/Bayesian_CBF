import numpy as np
from numpy import *

tensor = array

atan2 = arctan2


from_numpy = asarray

def to(arr, device=None, **kw):
    return arr.astype(**kw)

rand = np.random.rand

testing = np.testing
