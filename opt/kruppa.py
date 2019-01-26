#import numpy as np
import numpy as np
import autograd.numpy as anp
from autograd import jacobian
from utils import vmath as M
from utils import cv_wrap as W
from tf import transformations as tx
import cv2
from scipy.optimize import least_squares

class KruppaSolver(object):
    """
    Hartley's formulation
    https://ieeexplore.ieee.org/document/574792
    """
    def __init__(self):
        self.cache_ = {}
        self.params_ = dict(
            ftol=1e-5,
            xtol=1e-7,
            loss='linear',
            #x_scale='jac',
            max_nfev=1024,
            #method='lm',
            method='trf',
            verbose=2,
            #tr_solver='lsmr',
            tr_solver='exact',
            f_scale=100.0
            )

        self.jac = jacobian(self.err_anp)#, np=anp)

    def wrap_K(self, K):
        return K[(0,0,0,1,1),(0,1,2,1,2)]

    def unwrap_K(self, K, np=np):
        res = np.array([
            K[0], K[1], K[2],
            K[1], K[3], K[4],
            K[2], K[4], 1.0]).reshape(3,3)
        return res

    def A2K(self, A):
        return A.dot(A.T)

    def K2A(self, K):
        """ closed-form decomposition for {A | A.AT=K} """
        k1,k2,k3,k4,k5 = K[(0,0,0,1,1),(0,1,2,1,2)]
        #e00 = np.sqrt( k1 - k3**2 - (k2-k3*k5)**2 / (k4-k5**2) )
        e00 = np.sqrt((k1*k4 - k1*k5**2 - k2**2 + 2*k2*k3*k5 - k3**2*k4)/(k4 - k5**2))
        e01 = (k2 - k3*k5) / np.sqrt(k4-k5**2)
        if e01 < 0:
            e01 = -e01
        e02 = k3
        e11 = np.sqrt(k4-k5**2)
        e12 = k5
        return np.float32([e00,e01,e02,0,e11,e12,0,0,1]).reshape(3,3)

    def err_anp(self, K):
        e = self.err(K, np=anp)
        return e

    @staticmethod
    def mul3(a,b,c,np=np):
        # nx3 * 3x3 * nx3
        return np.einsum('...a,ab,...b->...',a,b,c,
                optimize=True)

    def err(self, K, np=np):
        # ( NOTE : K != cameraMatrix)
        K = self.unwrap_K(K, np=np)

        u1,u2,u3 = [self.cache_[k] for k in ['u1','u2','u3']]
        v1,v2,v3 = [self.cache_[k] for k in ['v1','v2','v3']]
        s1,s2    = [self.cache_[k] for k in ['s1','s2']]

        nmr1 = self.mul3(v2,K,v2,np=np)
        dmr1 = (s1*s1) * self.mul3(u1,K,u1,np=np)
        e1   = (nmr1 / dmr1)

        nmr2 = -self.mul3(v2,K,v1,np=np)
        dmr2 = (s1*s2) * self.mul3(u1,K,u2,np=np)
        e2   = (nmr2 / dmr2)

        nmr3 = self.mul3(v1,K,v1,np=np)
        dmr3 = (s2*s2) * self.mul3(u2,K,u2,np=np)
        e3   = (nmr3 / dmr3)

        err12 = (e1 - e2).ravel()
        err23 = (e2 - e3).ravel()

        return np.concatenate([err12, err23])
    
    def __call__(self, A, Fs):
        # A = camera Matrix

        # Fs = Nx3x3 Fundamental Matrix
        U, S, Vt = np.linalg.svd(Fs)
        u1, u2, u3 = U[...,:,0], U[...,:,1], U[...,:,2]
        v1, v2, v3 = Vt[...,0,:], Vt[...,1,:], Vt[...,2,:]
        s1, s2  = S[...,0], S[...,1]

        for k in ['u1','u2','u3','v1','v2','v3','s1','s2']:
            self.cache_[k] = vars()[k]

        #e = self.err(K)

        K = A.dot(A.T)
        res = least_squares(
                self.err, self.wrap_K(K),
                jac=self.jac,
                **self.params_
                )

        K = self.unwrap_K(res.x)
        #print 'K (optimized)'
        return self.K2A(K)

class KruppaSolverMC(object):
    """
    http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FUSIELLO3/node4.html
    ftp://mi.eng.cam.ac.uk/pub/reports/mendonca_self-calibration.pdf
    """
    def __init__(self):
        self.params_ = dict(
            ftol=1e-5,
            xtol=1e-7,
            loss='linear',
            #x_scale='jac',
            max_nfev=1024,
            #method='lm',
            method='trf',
            verbose=2,
            #tr_solver='lsmr',
            tr_solver='exact',
            f_scale=100.0
            )
        self.jac = jacobian(self.err_anp)#, np=anp)

    def wrap_A(self, A, np=np):
        #fx,fy,cx,cy
        return A[(0,0,0,1,1),(0,1,2,1,2)]#.ravel()

    def unwrap_A(self, A, np=np):
        return np.array([
            A[0],A[1],A[2],
            0,A[3],A[4],
            0,0,1]).reshape(3,3)

    def err_anp(self, params, Fs):
        e = self.err(params, Fs, np=anp)
        return e

    def err(self, params, Fs, np=np):
        A = self.unwrap_A(params[:4], np=np)
        Es = np.einsum('ba,...bc,cd->...ad', A, Fs, A)
        s = np.linalg.svd(Es,full_matrices=False, compute_uv=False)
        c = (s[..., 0] / s[...,1]) - 1.0
        return c

    def __call__(self, A, Fs):
        res = least_squares(
                self.err, self.wrap_A(A),
                args=(Fs,),
                jac=self.jac,
                x_scale='jac',
                **self.params_)
        A = self.unwrap_A(res.x)
        #print 'K (optimized)'
        #print A
        return A

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


def main():
    seed = np.random.randint( 65536 )
    #seed = 55507
    #seed = 34112

    print('seed', seed)
    np.random.seed( seed )

    #K = np.float32([500,0,320,0,500,240,0,0,1]).reshape(3,3)
    K = np.float32([1260,0,280,0,1260,230,0,0,1]).reshape(3,3)
    K0 = K.copy()

    s_noise = 100.0
    K0[0,0] = np.random.normal(K0[0,0], scale=s_noise) # fx
    K0[1,1] = np.random.normal(K0[1,1], scale=s_noise) # fy
    K0[0,2] = np.random.normal(K0[0,2], scale=s_noise) # cx
    K0[1,2] = np.random.normal(K0[1,2], scale=s_noise) # cy

    print 'K'
    print K
    print 'K0'
    print K0

    Fs = []
    for i in range(128):
        print '{}/{}'.format(i,128)
        p1, p2, x, P1, P2 = gen(min_n=64, K=K)
        F, _ = W.F(p1, p2,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=0.999,
                confidence=1.0
                )
        #print 'F', F
        #E, _ = cv2.findEssentialMat(p1,p2,K,
        #        cv2.FM_RANSAC,
        #        0.999,
        #        0.1)
        #        #thresh=1.0,
        #        #prob=0.999)
        #F = M.E2F(E, K=K)
        #print 'F', F / F[2,2]

        Fs.append(F)
    Fs = np.asarray(Fs, dtype=np.float64)

    # two-step refinement?
    K0 = KruppaSolverMC()(K0, Fs)
    print 'K', K
    K = KruppaSolver()(K0, Fs)
    print 'K', K

if __name__ == "__main__":
    main()
