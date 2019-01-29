import numpy as np
from tf import transformations as tx
import cv2
from utils import vmath as M
from utils import cv_wrap as W

def generate_data(
        n=100,
        max_it=128,
        K=None,
        T_c2b=None,
        s_pos=None,
        s_pt3=None,
        seed=None,
        ):
    """
    Generate Multiview Data. (Currently 2 Views)
    """
    if seed is not None:
        np.random.seed(seed)

    if s_pos is None:
        s_pos = [0.5, 0.2, np.deg2rad(30.0)]
    if s_pt3 is None:
        s_pt3 = 5.0

    # camera intrinsic parameters
    if K is None:
        # default K
        K = np.reshape([
            499.114583, 0.000000, 325.589216,
            0.000000, 498.996093, 238.001597,
            0.000000, 0.000000, 1.000000], (3,3))
    Ki = tx.inverse_matrix(K)

    # camera extrinsic parameters
    if T_c2b is None:
        T_c2b = tx.compose_matrix(
                        angles=[-np.pi/2-np.deg2rad(10),0.0,-np.pi/2],
                        translate=[0.174,0,0.113])
    T_b2c = tx.inverse_matrix(T_c2b)

    # generate pose
    pose = np.random.normal(scale=s_pos, size=3)
    x,y,h = pose

    # convert base_link pose to camera pose
    R = tx.euler_matrix(0, 0, h)[:3,:3]
    T_b2o = tx.compose_matrix(
            translate= (x,y,0),
            angles = (0,0,h)
            )
    T_o2b = tx.inverse_matrix(T_b2o)

    Tcc = np.linalg.multi_dot([
        T_b2c,
        T_o2b,
        T_c2b
        ])
    Rcc = Tcc[:3,:3]
    tcc = Tcc[:3,3:]

    # convert camera pose to OpenCV format (Rodrigues parametrized)
    rvec = cv2.Rodrigues(Rcc)[0]
    tvec = tcc.ravel()

    # generate landmark points
    # ensure points are valid
    res = {
            'pt3' : [],
            'pt2a': [],
            'pt2b': []
            }
    cnt = 0 # keep track of how many points were added to the dataset

    for i in range(max_it):
        # generate more points than requested, considering masks
        pt3 = np.random.normal(scale=s_pt3, size=(n, 3))
        pt3[:,2] = np.abs(pt3[:,2]) # forgot! positive depth required.

        # view 1 (identity)
        pt2a = cv2.projectPoints(
                pt3, np.zeros(3), np.zeros(3),
                cameraMatrix=K,
                distCoeffs=np.zeros(5)
                )[0][:,0]

        # view 2 (w/h offset)
        pt2b = cv2.projectPoints(
                pt3, rvec, tvec,
                cameraMatrix=K,
                distCoeffs = np.zeros(5)
                )[0][:,0]

        # filter with mask
        msk = np.logical_and.reduce([
            0 <= pt2a[:,0],
            pt2a[:,0] < 640,
            0 <= pt2a[:,1],
            pt2a[:,1] < 480,
            0 <= pt2b[:,0],
            pt2b[:,0] < 640,
            0 <= pt2b[:,1],
            pt2b[:,1] < 480,
            ])

        # add data
        res['pt3'].append(pt3[msk])
        res['pt2a'].append(pt2a[msk])
        res['pt2b'].append(pt2b[msk])

        cnt += msk.sum()

        if cnt >= n:
            break

    pt3 = np.concatenate(res['pt3'], axis=0)[:n]
    pt2a = np.concatenate(res['pt2a'], axis=0)[:n]
    pt2b = np.concatenate(res['pt2b'], axis=0)[:n]

    return pt3, pt2a, pt2b, pose

def _gen(max_n=100, min_n=16,
        w=640, h=480,
        K=None, Ki=None,
        ):
    if Ki is None:
        Ki = np.linalg.inv(K)

    p1 = np.random.uniform((0,0), (w,h), size=(max_n,2))
    d  = np.random.uniform(0.01, 100.0, size=(max_n,))
    #d[:] = 1.0 # planar scene
    x = d[:,None] * np.einsum('ab,...b->...a', Ki, M.to_h(p1))
    P1 = np.eye(3,4)

    rxn = np.random.uniform(-np.pi, np.pi, size=3)
    txn = np.random.uniform(-np.pi, np.pi, size=3)
    #txn *= 0.01#0.00000000001
    #print 'txn', txn
    #print 'rxn', rxn
    #print 'u-tx', M.uvec(txn)

    P2 = tx.compose_matrix(
            angles=rxn, translate=txn)[:3]

    p2h = np.einsum('ab,bc,...c->...a',
            K, P2, M.to_h(x))
    p2 = M.from_h(p2h)

    msk = np.logical_and.reduce([
        (p2h[..., -1] >= 0),
        0 <= p2[...,0],
        0 <= p2[...,1],
        p2[...,0] < w,
        p2[...,1] < h,
        ])
    p1, p2, x = [e[msk] for e in [p1,p2,x]]
    if len(p1) < min_n:
        # retry
        #return gen(max_n,min_n,w,h,K,Ki)
        return None
    return p1, p2, x, P1, P2

def gen(*args, **kwargs):
    while True:
        res = _gen(*args, **kwargs)
        if res is not None:
            return res
