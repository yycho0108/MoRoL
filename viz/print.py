import numpy as np
from tf import transformations as tx

def print_Rt(R, t, round=2):
    print '\tR', np.round(np.rad2deg(tx.euler_from_matrix(R)), round)
    print '\tt', np.round(t.ravel() / np.linalg.norm(t), round)

def print_ratio(a, b):
    pass
