import numpy as np
from autograd import numpy as anp
from autograd import jacobian
from scipy.optimize import least_squares
from core.calib.kruppa.common import mul3

class KruppaSolver(object):
    """
    Hartley's formulation
    https://ieeexplore.ieee.org/document/574792
    """
    def __init__(self, verbose=2):
        self.cache_ = {}
        self.params_ = dict(
            ftol=1e-10,
            xtol=1e-9,
            #gtol=1e-16,
            loss='linear',
            max_nfev=1024,
            method='trf',
            #method='lm',
            verbose=verbose,
            #tr_solver='lsmr',
            tr_solver='exact',
            #f_scale=1.0
            )

        self.jac = jacobian(self.err_anp)#, np=anp)

    def wrap_K(self, K):
        return K[(0,0,1,1),(0,2,1,2)]
        #return K[(0,0,0,1,1),(0,1,2,1,2)]

    def unwrap_K(self, K, np=np):
        #res = np.array([
        #    K[0], K[2], K[2],
        #    K[1], K[3], K[4],
        #    K[2], K[4], 1.0]).reshape(3,3)
        res = np.array([
            K[0], K[1]*K[3], K[1],
            K[1]*K[3], K[2], K[3],
            K[1], K[3], 1.0]).reshape(3,3)
        return res

    def A2K(self, A):
        return A.dot(A.T)

    def K2A(self, K):
        """ closed-form decomposition for {A | A.AT=K} """
        # M = A.AT, A = upper triangular
        # M.T = A.AT

        #k1,k2,k3,k4,k5 = K[(0,0,0,1,1),(0,1,2,1,2)]
        k1,k3,k4,k5 = K[(0,0,1,1),(0,2,1,2)]
        k2 = k3 * k5
        #print np.linalg.eig(K)
        #print cholesky(K)

        print 'k4', k4
        print 'k5', k5

        if k4 - k5 ** 2 < 0:
            return None
        if k1 - k3**2 - (k2-k3*k5)**2 / (k4-k5**2)  < 0:
            return None

        tmp = np.sqrt(k4 - k5 ** 2)#np.abs( np.lib.scimath.sqrt(k4 - k5 ** 2) )
        #e00 = np.sqrt((k1*k4 - k1*k5**2 - k2**2 + 2*k2*k3*k5 - k3**2*k4)/(k4 - k5**2))
        e00 = np.sqrt( k1 - k3**2 - (k2-k3*k5)**2 / (k4-k5**2) )
        e01 = (k2 - k3*k5) / tmp# np.sqrt(k4-k5**2)
        #if e01 < 0:
        #    e01 = -e01
        e02 = k3
        e11 = tmp
        e12 = k5

        return np.float32([e00,e01,e02,0,e11,e12,0,0,1]).reshape(3,3)

    def err_anp(self, K):
        e = self.err(K, np=anp)
        return e

    def err(self, K, np=np):
        # ( NOTE : K != cameraMatrix)
        K = self.unwrap_K(K, np=np)

        u1,u2,u3 = [self.cache_[k] for k in ['u1','u2','u3']]
        v1,v2,v3 = [self.cache_[k] for k in ['v1','v2','v3']]
        s1,s2    = [self.cache_[k] for k in ['s1','s2']]
        Ws = self.cache_['Ws']

        nmr1 = mul3(v2,K,v2,np=np)
        dmr1 = (s1*s1) * mul3(u1,K,u1,np=np)
        e1   = (nmr1 / dmr1)

        nmr2 = -mul3(v2,K,v1,np=np)
        dmr2 = (s1*s2) * mul3(u1,K,u2,np=np)
        e2   = (nmr2 / dmr2)

        nmr3 = mul3(v1,K,v1,np=np)
        dmr3 = (s2*s2) * mul3(u2,K,u2,np=np)
        e3   = (nmr3 / dmr3)

        #err12 = nmr1 * dmr2 - nmr2 * dmr1
        #err23 = nmr2 * dmr3 - nmr3 * dmr2
        #err31 = nmr3 * dmr1 - nmr1 * dmr3

        err12 = ((e1 - e2)).ravel()
        err23 = ((e2 - e3)).ravel()
        err31 = ((e1 - e3)).ravel()

        return np.concatenate([err12, err23, err31])

    #def err_USV(self, USV, np=np):
    #    K = self.cache_['K']
    #    u1,u2,u3 = USV[...,:3*3].reshape(-1,3,3)
    #    s1,s2,s3 = USV[...,3*3:-3*3].reshape(-1,3)
    #    v1,v2,v3 = USV[...,-3*3:].reshape(-1,3,3)

    #    nmr1 = mul3(v2,K,v2,np=np)
    #    dmr1 = (s1*s1) * mul3(u1,K,u1,np=np)
    #    e1   = (nmr1 / dmr1)

    #    nmr2 = -mul3(v2,K,v1,np=np)
    #    dmr2 = (s1*s2) * mul3(u1,K,u2,np=np)
    #    e2   = (nmr2 / dmr2)

    #    nmr3 = mul3(v1,K,v1,np=np)
    #    dmr3 = (s2*s2) * mul3(u2,K,u2,np=np)
    #    e3   = (nmr3 / dmr3)

    #    err12 = (e1 - e2).ravel()
    #    err23 = (e2 - e3).ravel()
    #    err31 = (e3 - e1).ravel()
    #    return np.concatenate([err12, err23, err31])
    
    def __call__(self, A, Fs, Ws):
        # A = camera Matrix

        # strum's focal length approximation
        #gl, gr = np.eye(3), np.eye(3)
        #gl[2,0] = 320.0
        #gl[2,1] = 240.0
        #gr[0,2] = 320.0
        #gr[1,2] = 240.0 # TODO : make this not hardcoded
        #Gs = np.einsum('ab,...bc,cd',gl,Fs,gr)
        #U, S, Vt = np.linalg.svd(Gs)
        #u1, u2, u3 = U[...,:,0], U[...,:,1], U[...,:,2]
        #v1, v2, v3 = Vt[...,0,:], Vt[...,1,:], Vt[...,2,:]
        #s1, s2  = S[...,0], S[...,1]
        #f = solve_focal(s1, s2, u1, u2, u3, v1, v2, v3)
        #print 'focal estimate', f
        #A = A.copy()
        #A[0,0] = A[1,1] = f

        # Fs = Nx3x3 Fundamental Matrix
        U, S, Vt = np.linalg.svd(Fs)
        u1, u2, u3 = U[...,:,0], U[...,:,1], U[...,:,2]
        v1, v2, v3 = Vt[...,0,:], Vt[...,1,:], Vt[...,2,:]
        s1, s2  = S[...,0], S[...,1]

        for k in ['u1','u2','u3','v1','v2','v3','s1','s2']:
            self.cache_[k] = vars()[k]
        self.cache_['Ws'] = Ws

        K = self.A2K(A)
        #e = self.err(K)
        #print 'e', e
        #print 'A', A
        #A2 = self.K2A(K)
        #print 'K', K
        print 'k-params', self.wrap_K(K)
        #print 'K-rec', self.unwrap_K(self.wrap_K(K))
        #print 'verify', A2

        res = least_squares(
                self.err,
                self.wrap_K(K),
                #x_scale=np.abs( self.wrap_K(K) ),
                x_scale='jac',
                jac=self.jac,
                **self.params_
                )

        K = self.unwrap_K(res.x)
        #print 'K (optimized)'
        return self.K2A(K)
