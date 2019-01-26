from scipy.linalg import eig, eigvals, ordqz, null_space
import numpy as np
from utils import vmath as M
from utils import cv_wrap as W
from tf import transformations as tx
import cv2
from matplotlib import pyplot as plt
from scipy.special import huber

from opt.ransac import RANSACModel

def dmodel(l, x):
    r2 = np.square(x).sum(axis=-1,keepdims=True)
    return x * (1.0 + l * r2)

def dmodel_i(l, x):
    r2 = np.square(x).sum(axis=-1,keepdims=True)
    x = x / (1.0 + l * r2)
    return x

def sampson_error(pt1, pt2, F):
    # 11.9 from H-Z
    # numerator
    nmr = np.square( np.einsum('...a,ab,...b->...', pt2, F, pt1) )

    dmr1 = np.einsum('ab,...b->...a', F, pt1)
    dmr1 = np.square(dmr1[...,:2]).sum(axis=-1)

    dmr2 = np.einsum('ab,...b->...a', F.T, pt2)
    dmr2 = np.square(dmr2[...,:2]).sum(axis=-1)

    return nmr / (dmr1 + dmr2)

def linv(A):
    return np.linalg.inv(A.T.dot(A)).dot(A.T)

class CoDSolver(object):
    def __init__(self):
        pass
    
class DistSolver(object):
    def __init__(self, w, h):
        self.w_ = w
        self.h_ = h
        self.l_ = 0.0
        self.rsc_ = RANSACModel(
                n_model=9,
                model_fn=self.rsc_model,
                err_fn=self.rsc_err,
                thresh=(1e-1/(w+h))**2, # accept ~.01 threshold in pixel space
                prob=0.999
                )
        self.cache_ = {}

    def rsc_model(self, idx):
        D1, D2, D3 = [self.cache_[e] for e in ['D1','D2','D3']]
        l, F = self._solve(D1[idx], D2[idx], D3[idx])
        return [l, F, 0]

        #l, F = model

        #if len(l) <= 0:
        #    return None

        #i = np.argsort(l)[len(l)//2]
        #return (l[i], F[i])

        # NOTE : hack not to deal with crap
        #return np.median( model )

    def rsc_err(self, model):
        # unroll data
        p1, p2 = [self.cache_[e] for e in ['p1','p2']]
        l, F, _ = model

        if len(l) <= 0:
            return np.full(len(p1), np.inf)

        # TODO : compute error only on inliers?
        best_err = np.full(len(p1), np.inf)
        best_err_ = np.inf # avg

        for i, (l_, F_) in enumerate(zip(l,F)):
            up1 = dmodel_i(l_, p1)
            up2 = dmodel_i(l_, p2)
            err = sampson_error(M.to_h(up1), M.to_h(up2), F_)
            err_ = err.mean() # avg

            if err_ < best_err_:
                # update selection index
                best_err = err
                best_err_ = err_
                model[-1] = i

        return best_err

    def norm_fw(self, x):
        w,h = self.w_, self.h_
        return (x - [[w/2., h/2.]]) / (w + h)

    def norm_bw(self, x):
        w,h = self.w_, self.h_
        return (x * (w+h)) + [[w/2., h/2.]]

    def _solve(self, D1, D2, D3):
        # D1,D2,D3 Nx9
        if D1.shape != (9,9):
            D1_s = D1.T.dot(D1)
            D2_s = D1.T.dot(D2)
            D3_s = D1.T.dot(D3)
        else:
            D1_s = D1
            D2_s = D2
            D3_s = D3

        Z = np.zeros((9,9), dtype=D1.dtype)
        I = np.eye(9)

        A = np.block([
            [-D1_s, Z],
            [Z, I]])
        B = np.block([
            [D2_s, D3_s],
            [I, Z]])


        #print 'A', A
        #print 'B', B

        #pre = np.zeros((18,18), dtype=D1.dtype)
        #pre[:9,:9] = -np.linalg.pinv(D1_s)
        #pre[9:,9:] = np.eye(9)
        #pre = linv(A)

        #print 'sbz', pre.dot(A).sum() - np.diag( pre.dot(A) ).sum()

        #l, _ = np.linalg.eig( pre.dot(B) )
        #l = 1.0 / l

        l, vr = eig(A, B, left=False, right=True) # A.Vr = W.B.Vr
        #l = eigvals(A, B)
        #_, _, alpha, beta, _, _ = ordqz(A, B, output='real')
        #l = (alpha / beta)[:9]

        #print 'hmm'
        #print l[:9]
        #print l[9:]
        #print 'l', l.shape

        msk = np.logical_and.reduce([
            #np.not_equal(alpha, 0),
            #np.not_equal(beta, 0),

            np.isfinite( l ),
            np.isreal( l ),
            np.not_equal(l, 0),

            np.all(np.isfinite(vr), axis=-1),
            np.all(np.isreal(vr), axis=-1),

            np.less(np.abs(l), 10.0)
            ])
        sel = np.where(msk)[0]
        l = np.real( l[sel] )
        F = None

        if l.size > 0:
            S = D1_s[None,...] + l[:, None,None] * D2_s[None,...] + l[:, None, None]**2 * D3_s[None,...]
            # vectorized nullspace
            _, w, Vt = np.linalg.svd(S)
            F = Vt[np.arange(len(l)), np.argmin(w, axis=-1)].reshape(-1,3,3)
            F /= F[..., -1:, -1:]

            # individual nullspace
            #f = null_space(S).reshape(3,3)
            #F = f / f[..., -1:,-1:]

        return l, F

    def _no_ransac(self):
        # copied here for archival purposes. DO NOT USE
        ls = []
        D1, D2, D3, p1, p2 = [self.cache_[k] for k in 'D1,D2,D3,p1,p2'.split(',')]
        n = len(D1)

        ml = self._solve(D1,D2,D3)[0]
        print ml

        for _ in range( max(n / 9, 1) ):
            idx = np.random.choice(n, 128, replace=False)
            l, F = self._solve(D1[idx], D2[idx], D3[idx])
            ls.extend(l)
        ls = np.array(ls, dtype=np.float64)
        #print 'sub',
        print ls
        #plt.hist(ls)
        #plt.show()
        #lo = np.percentile(ls, 20)
        #hi = np.percentile(ls, 80)
        #return ls[np.logical_and(lo<=ls, ls<hi)].mean()

        #print ls
        return np.median(ls)

    def undistort(self, p):
        p = self.norm_fw(p)
        p = dmodel_i(self.l_, p)
        p = self.norm_bw(p)
        return p

    def __call__(self, p1, p2, max_it=256):
        # epipolar constraint
        # p'^T.F.p = 0

        # change of coordinates
        p1 = self.norm_fw(p1)
        p2 = self.norm_fw(p2)

        # NOTE : lambda must be corrected by the actual focal length
        # when that gets determined, I guess

        x1, y1 = p1[...,0], p1[...,1]
        r1_sq = np.square(x1) + np.square(y1)
        x2, y2 = p2[...,0], p2[...,1]
        r2_sq = np.square(x2) + np.square(y2)

        c0 = np.zeros_like(x1)
        c1 = np.ones_like(x1)

        D1 = np.array([
            x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, c1],
            dtype=p1.dtype).T
        D2 = np.array([
            c0, c0, x2*r1_sq, c0, c0, y2*r1_sq, x1*r2_sq, y1*r2_sq, r1_sq+r2_sq],
            dtype=p1.dtype).T
        D3 = np.array([
            c0, c0, c0, c0, c0, c0, c0, c0, r1_sq*r2_sq],
            dtype=p1.dtype).T
        # (D1 + l*D2 + l**2*D3)f = 0
        # generalized form
        # A V = w B V
        # == A=D3 B=D2, C=D1
        #Bmat = [[B,A],[I,0]]
        #Amat = [[-C,I]]
        
        self.cache_['D1'] = D1
        self.cache_['D2'] = D2
        self.cache_['D3'] = D3
        self.cache_['p1'] = p1
        self.cache_['p2'] = p2

        #l = self._no_ransac()
        #self.l_ = l
        #return 1, l, np.ones(len(p1), dtype=np.bool)
        n_it, res = self.rsc_(len(p1), max_it)
        if res is not None:
            m_l, m_F, m_i = res['model']
            self.l_ = m_l[m_i]
            return n_it, m_l[m_i], res['inl']
        else:
            return 0, self.l_, 0

def gen(max_n=100, min_n=16,
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
        return gen(max_n,min_n,w,h,K,Ki)
    return p1, p2, x, P1, P2


def main():
    np.set_printoptions(5)
    seed = np.random.randint( 65536 )
    #seed = 13863
    #seed = 21942
    #seed = 14549
    #seed = 24814
    print 'seed', seed
    np.random.seed(seed)
    w, h = (640, 480)

    l = np.random.uniform(-0.2, 0.2)

    K = np.float32([
        500,0,320,
        0,500,240,
        0,0,1]).reshape(3,3)
    #K = np.eye(3)

    D = np.zeros(5)
    D[0] = l

    solver = DistSolver(w,h)
    p1, p2, x, P1, P2 = gen(max_n=1024, min_n=256,
            w=w, h=h,
            K=K)

    print 'F'
    print W.F(p1, p2)[0]
    E = W.E(p1, p2, cameraMatrix=K)[0]
    print 'E'
    print E

    # distort through OpenCV
    Ki = np.linalg.inv(K)
    dp1 = W.project_points((M.to_h(np.random.normal(loc=p1, scale=0.2)).dot(Ki.T)),
            rvec=np.zeros(3), tvec=np.zeros(3),
            cameraMatrix=K,
            distCoeffs=D)
    dp2 = W.project_points((M.to_h(np.random.normal(loc=p2, scale=0.2)).dot(Ki.T)),
            rvec=np.zeros(3), tvec=np.zeros(3),
            cameraMatrix=K,
            distCoeffs=D)

    #print 'p1', p1[0]
    #print 'dp1', dp1[0]

    l2 = solver(dp1, dp2)
    print ' === results === '
    print 'orig', l
    print 'n_it', l2[0]
    print 'dist (raw)', l2[1]
    print 'dist (focal-corrected)', l2[1] * (K[1,1] / (w+h)) **2
    print 'n_in', l2[2].sum() / float(l2[2].size)

    p1_r = solver.undistort( dp1 )
    print 'distortion mean error', np.linalg.norm(dp1 - p1, axis=-1).mean()
    print 'mean error', np.linalg.norm(p1_r - p1, axis=-1).mean()

    p2_r = solver.undistort( dp2 )
    print 'distortion mean error', np.linalg.norm(dp2 - p2, axis=-1).mean()
    print 'mean error', np.linalg.norm(p2_r - p2, axis=-1).mean()

    plt.plot(p1[:,0], p1[:,1], 'r+', label='orig')
    plt.plot(dp1[:,0], dp1[:,1], 'bx', label='dist')
    plt.plot(p1_r[:,0], p1_r[:,1], 'g.', label='rec')
    plt.plot([0,w,w,0,0],[0,0,h,h,0], 'k--')
    plt.xlim([-w/4., w + w/4.])
    plt.ylim([-h/4., h + h/4.])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
