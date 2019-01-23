import numpy as np
from tf import transformations as tx

def print_Rt(R, t, round=2):
    print '\tR', np.round(np.rad2deg(tx.euler_from_matrix(R)), round)

    n = np.linalg.norm(t)
    u = (t.ravel()/n  if n > np.finfo(np.float32).eps else t.ravel())
    print '\tt', np.round(u, round)

def print_ratio(a, b):
    pass
