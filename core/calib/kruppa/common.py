#import numpy as np
import numpy as np
import autograd.numpy as anp
from autograd import jacobian
from utils import vmath as M
from utils import cv_wrap as W
from tf import transformations as tx
import cv2
from scipy.optimize import least_squares
from scipy.linalg import cholesky
from scipy.linalg import lu as LU

from matplotlib import pyplot as plt

def mul3(a,b,c,np=np):
    # nx3 * 3x3 * nx3
    return np.einsum('...a,ab,...b->...',a,b,c,
            optimize=True)

def stacked_lstsq(L, b, rcond=1e-10):
    """
    Solve L x = b, via SVD least squares cutting of small singular values
    L is an array of shape (..., M, N) and b of shape (..., M).
    Returns x of shape (..., N)
    """
    u, s, v = np.linalg.svd(L, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond*s_max
    inv_s = np.zeros_like(s)
    inv_s[s >= s_min] = 1/s[s>=s_min]

    tmp = np.einsum('...ji,...j->...i', u, b.conj())
    x = np.einsum('...ji,...j->...i', v,
                  inv_s * tmp)
    return np.conj(x, x)

def svd_jac(u,s,vt):
    # for MxN matrix X,
    # i = {0..M-1], j={0..N-1}
    # k = {0..N-1}, l = {i+1..N-1}

    # u = presumably MxB
    # v = presumably BxM

    v = vt.swapaxes(-1,-2)

    A = []
    b = []
    idx = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3): # ??
                    idx.append( (i,j,k,l) )
                    A.append([
                        [s[...,l],s[...,k]],
                        [s[...,k],s[...,l]]
                        ])
                    b.append([
                        u[...,i,k] * v[...,j,l],
                        -u[...,i,l] * v[...,j,k]
                        ])

    A = np.asarray(A)
    b = np.asarray(b)

    A = A.transpose(3,0,1,2) # N,27?,2,2
    b = b.transpose(2,0,1) # N,27?,2,1

    W = stacked_lstsq(A, b)
    Wu = np.zeros(shape=s.shape[:-1] + (3,3,3,3))
    Wv = np.zeros(shape=s.shape[:-1] + (3,3,3,3))

    i,j,k,l = zip(*idx)

    Wu[:,i,j,k,l] = W[:,:,0]
    Wv[:,i,j,k,l] = W[:,:,1]

    du_daij = np.einsum('...ab,...ijbc->...acij', u, Wu)
    ds_daij = np.einsum('...ac,...bc->...cab', u, v)
    dv_daij = np.einsum('...ab,...ijbc->...acij', -v, Wv) # Note : dV/dA; NOT dVt/dA

    return du_daij, ds_daij, dv_daij
