import numpy as np
from tf import transformations as tx

def rint(x):
    return np.round(x).astype(np.int32)

def anorm(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def inv(x):
    return np.linalg.inv(x)

def to_h(x, z=False):
    x = np.asarray(x)
    o = np.ones_like(x[..., :1])
    if z:
        o *= 0
    return np.concatenate([x,o], axis=-1)

def from_h(x):
    return x[...,:-1] / x[...,-1:]

def norm(x, *args, **kwargs):
    return np.linalg.norm(x, *args, axis=-1, **kwargs)

def uvec(x):
    n = norm(x, keepdims=True)
    if n < np.finfo(np.float32).eps:
        return x
    else:
        return x / norm(x, keepdims=True)

def R2(x, T=None, c=None, s=None):
    if T is None:
        shape = tuple(np.shape(x)[:-1]) + (2,2)
        T = np.zeros(shape, dtype=np.float32)
    if c is None:
        c = np.cos(x)
    if s is None:
        s = np.sin(x)
    T[...,0,0] = c
    T[...,0,1] = -s
    T[...,1,0] = s
    T[...,1,1] = c

    return T

def Rz(x, T=None, c=None, s=None):
    if T is None:
        shape = tuple(np.shape(x)[:-1]) + (4,4)
        T = np.zeros(shape, dtype=np.float32)
    if c is None:
        c = np.cos(x)
    if s is None:
        s = np.sin(x)

    T[..., 0, 0] = c
    T[..., 0, 1] = -s
    T[..., 1, 0] = s
    T[..., 1, 1] = c
    T[..., 2, 2] = 1

    return T

def Ti(T):
    """
    T = tx.compose_matrix(angles=np.random.uniform(-np.pi,np.pi,3), translate=np.random.uniform(-np.pi,np.pi,3))
    print T
    print Ti(T).dot(T)
    print Ti(Ti(T))
    """
    R = T[..., :3,:3]
    t = T[..., :3,3:]

    res = np.zeros_like(T)
    res[..., :3, :3] = R.swapaxes(-1, -2)
    res[..., :3, 3:] = -np.einsum('...ba,...bc->...ac', R, t)
    res[..., 3, 3] = 1

    return res

def Rti(R, t):
    Ri = R.swapaxes(-1, -2)
    if np.ndim(ti) < Ri:
        # case (...,D)
        ti = -np.einsum('...ab,...b->...a', RT, t)
    else:
        # case (...,D,1)
        ti = -np.einsum('...ab,...bc->...ac', RT, t)
    return Ri, ti

def lerp(a,b,w):
    return (a * (1.0-w)) + (b*w)

def flerp(a,b,w,f,fi):
    return fi(lerp(f(a), f(b), w))

def rlerp(ra, rb, w):
    Ra = np.eye(4, dtype=np.float32)
    Rb = np.eye(4, dtype=np.float32)
    Ra[:3,:3] = ra
    Rb[:3,:3] = rb

    qa = tx.quaternion_from_matrix(Ra)
    qb = tx.quaternion_from_matrix(Rb)
    q = tx.quaternion_slerp(q0, q1, w)
    R = tx.quaternion_matrix(q)[:3,:3]
    return R

def add_p3(p3, dp3):
    x = p3[..., 0]
    y = p3[..., 1]
    h = p3[..., 2]
    dx = dp3[...,0]
    dy = dp3[...,1]
    dh = dp3[...,2]

    dp = np.einsum('...ab,...b->...a',
            R2(h), dp3[...,:2])
    h = anorm(h+dh)[..., None]

    res = np.concatenate([dp,h], axis=-1)
    return res

def p3_T(p3):
    """
    Computes homogeneous transform,
    from p3=(x,y,h) parametrization.
    The resultant transform represents a conversion
    from the p3 frame to the `world` frame.
    """
    p3 = np.asarray(p3)

    if np.ndim(p3) > 1:
        shape = tuple(p3.shape[:-1]) + (4,4)
        T = np.zeros(shape, dtype=np.float32)
        # translation
        T[...,0,3] = p3[...,0]
        T[...,1,3] = p3[...,1]
        # rotation
        Rz(p3[...,2], T=T)
        # homogeneous
        T[...,3,3] = 1
        return T
    else:
        x, y, h = p3
        return tx.compose_matrix(
                translate=(x,y,0),
                angles=(0,0,h))
    pass

def rx3(R, x):
    rx = np.einsum('...ab,...b->...a', T[..., :3,:3], x)
    return rx

def tx3(T, x):
    rx = np.einsum('...ab,...b->...a', T[..., :3,:3], x)
    return rx + T[..., :3,3:].swapaxes(-2,-1)

def rtx3(r, t, x):
    return x.dot(r.swapaxes(-2,-1)) + t

def tx4(T, x):
    return np.einsum('...ab,...b->...a', T, x)

def ands(*args, **kwargs):
    return np.logical_and.reduce(*args, **kwargs)

def intersect2d(a, b):
    """
    from https://stackoverflow.com/a/8317403
    """
    nrows, ncols = a.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
        'formats':ncols * [a.dtype]}
    c = np.intersect1d(a.view(dtype), b.view(dtype))
    c = c.view(a.dtype).reshape(-1, ncols)
    return c

def robust_mean(x, margin=10.0, weight=None):
    if len(x) <= 0:
        return np.nan
    x = np.asarray(x, dtype=np.float32)
    s_lo = np.percentile(x, 50.0 - margin)
    s_hi = np.percentile(x, 50.0 + margin)
    msk = np.logical_and.reduce([
        s_lo <= x, x <= s_hi
        ])

    if weight is None:
        return x[msk].mean()
    else:
        # compute normalized weight
        #print 'weighting'
        #print weight.shape
        #print msk.shape
        w = weight[msk[...,0]]
        w = w / w.sum()

        return np.sum(x[msk] * w, axis=-1)

def invert_index(i, n):
    msk = np.ones(n, dtype=np.bool)
    msk[i] = False
    return np.where(msk)[0]

def E2F(E, K=None, Ki=None):
    if Ki is None:
        Ki = inv(K)
    F = np.linalg.multi_dot([
        Ki.T, E, Ki])
    # normalize
    #F /= F[-1,-1]
    return F

def F2E(F, K=None, Ki=None):
    if K is None:
        K = inv(Ki)
    return np.linalg.multi_dot([
        K.T, F, K])

def jac_to_cov(J, e=None, n_params=4):
    """ from scipy/optimize/minpack.py#L739 """

    # naive version
    #JT = J.swapaxes(-1,-2)
    #JTJ = np.einsum('...ab,...bc->...ac', JT, J)
    #cov = np.linalg.pinv(JTJ)

    _, s, VT = np.linalg.svd(J, full_matrices=False)
    thresh = np.finfo(np.float32).eps * max(J.shape) * s[0]
    s = s[s > thresh]
    VT = VT[:s.size]

    cov = np.dot(VT.T / s**2, VT)
    if e is not None:
        # WARNING : # of params assumed and hardcoded to be 4
        rhs = np.square(e).sum() / float(len(e) - n_params)
        cov = cov * rhs
    return cov
