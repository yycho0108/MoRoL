import numpy as np
import cv2
from tf import transformations as tx

from utils import vmath as M
from utils import cv_wrap as W
from viz.log import print_Rt
from core.common import recover_pose, recover_pose_perm

def score_H(pt1, pt2, H, sigma=1.0):
    """ Homography model symmetric transfer error. """
    score = 0.0
    th = 5.991 # ??? TODO : magic number
    iss = (1.0 / (sigma*sigma))

    Hi = np.linalg.inv(H)
    pt2_r = M.from_h(M.to_h(pt1).dot(H.T))
    pt1_r = M.from_h(M.to_h(pt2).dot(Hi.T))

    e1 = np.square(pt1 - pt1_r).sum(axis=-1)
    e2 = np.square(pt2 - pt2_r).sum(axis=-1)

    #score = 1.0 / (e1.mean() + e2.mean())
    chi_sq1 = e1 * iss
    msk1 = (chi_sq1 <= th)
    score += ((th - chi_sq1) * msk1).sum()

    chi_sq2 = e2 * iss
    msk2 = (chi_sq2 <= th)
    score += ((th - chi_sq2) * msk2).sum()
    return score, (msk1 & msk2)

def sampson_error(pt1, pt2, F):
    # 11.9 from H-Z
    # numerator
    nmr = np.square( np.einsum('...a,ab,...b->...', pt2, F, pt1) )

    dmr1 = np.einsum('ab,...b->...a', F, pt1)
    dmr1 = np.square(dmr1[...,:2]).sum(axis=-1)

    dmr2 = np.einsum('ab,...b->...a', F.T, pt2)
    dmr2 = np.square(dmr2[...,:2]).sum(axis=-1)

    return nmr / (dmr1 + dmr2)

def sampson_H(pt1, pt2, H):
    # err = H.dot(
    # symt = Hi(pt2) - pt1
    # repj = pt1 - Hi(pt2)
    # should be the same??
    pass

def score_F(pt1, pt2, F, sigma=1.0):
    """
    Fundamental Matrix symmetric transfer error.
    reference:
        https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fundam.cpp#L728
    """
    score = 0.0
    th = 3.841 # ??
    th_score = 5.991 # ?? TODO : magic number
    iss = (1.0 / (sigma*sigma))

    pt1_h = M.to_h(pt1)
    pt2_h = M.to_h(pt2)

    x1, y1 = pt1.T
    x2, y2 = pt2.T

    a, b, c = pt1_h.dot(F.T).T # Nx3
    s2 = 1./(a*a + b*b);
    d2 = a * x2 + b * y2 + c
    e2 = d2*d2*s2

    a, b, c = pt2_h.dot(F).T
    s1 = 1./(a*a + b*b);
    d1 = a * x1 + b * y1 + c
    e1 = d1*d1*s1

    print 'e1', e1[:5]
    print 'e2', e2[:5]
    #e1r = np.einsum('...a,ab,...b->...', pt2_h, F, pt1_h)
    #print 'e1r', e1r[0]

    #print 'sampson'
    #print sampson_error(pt1_h,pt2_h,F)
    #print (e1+e2) / 4.0

    #e = 2 * sampson_error(pt1_h,pt2_h,F)
    ##e = (e1+e2) / 2.0
    #ek = e*iss
    #print 'hmm', (th_score - ek)[ek<=th].sum() * 2

    #score = 1.0 / (e1.mean() + e2.mean())
    chi_sq2 = e2 * iss
    msk2 = (chi_sq2 <= th)
    score += ((th_score - chi_sq2) * msk2).sum()

    chi_sq1 = e1* iss
    msk1 = (chi_sq1 <= th)
    score += ((th_score - chi_sq1) * msk1).sum()

    return score, (msk1 & msk2)

def gen(max_n=100, min_n=16,
        w=640, h=480,
        K=None, Ki=None
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
    txn *= 0.01#0.00000000001
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
        return gen(max_n,min_n,w,h,K,Ki)
    return p1, p2, x, P1, P2

def main():
    #np.set_printoptions(3)
    #seed = 47748
    seed = np.random.randint(65536)
    print('seed', seed)
    np.random.seed( seed )

    n = 128
    w,h = (640,480)
    K = np.float32([
        500, 0, w/2.0,
        0, 500, h/2.0,
        0,   0, 1]).reshape(3,3)
    p1, p2, x, P1, P2 = gen(max_n=n,w=w,h=h,K=K)

    print 'P2'
    print P2
    print_Rt( P2[:,:3], P2[:,3:] )

    p1 = np.random.normal(p1, scale=0.5)
    p2 = np.random.normal(p2, scale=0.5)

    F, _ = W.F(p1, p2)#, cameraMatrix=K)
    E, _ = W.E(p1, p2, cameraMatrix=K)

    #print 'validation ...'
    #print 'E', E
    #print 'F', F
    #print 'E->F', M.E2F(E, K=K)
    #print 'F->E', M.F2E(F, K=K)

    _, R, t, _, _ = recover_pose(E, K, p1, p2,
            log=False
            )
    print 'E', E
    print 'ER', R
    print 'Et', t
    print_Rt(R, t)
    H, _ = W.H(p1, p2, method=cv2.RHO)

    res_h, Hr, Ht, Hn = cv2.decomposeHomographyMat(H, K)
    print [M.uvec(e.ravel()) for e in Ht]
    _, Hr, Ht, _, _ = recover_pose_perm(zip(Hr, Ht), K, p1, p2,
            log=False
            )

    print 'Hr', Hr
    print 'Ht', Ht
    print_Rt(Hr, Ht)
    print 'H', H

    sf, _ = score_F(p1, p2, F)
    sh, _ = score_H(p1, p2, H)

    rh = (sh / (sh + sf))

    print 'sf', sf
    print 'sh', sh
    print 'rh', rh


if __name__ == "__main__":
    main()
