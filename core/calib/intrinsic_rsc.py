from opt.ransac import RANSACModel
from core.calib.kruppa import KruppaSolver, KruppaSolverMC
from core.calib.esv import ESVSolver
import numpy as np
from core.calib.kruppa.common import mul3
from matplotlib import pyplot as plt

#def sampson_error(pt1, pt2, F):
#    # 11.9 from H-Z
#    # numerator
#    nmr = np.square( np.einsum('...a,ab,...b->...', pt2, F, pt1) )
#
#    dmr1 = np.einsum('ab,...b->...a', F, pt1)
#    dmr1 = np.square(dmr1[...,:2]).sum(axis=-1)
#
#    dmr2 = np.einsum('ab,...b->...a', F.T, pt2)
#    dmr2 = np.square(dmr2[...,:2]).sum(axis=-1)
#
#    return nmr / (dmr1 + dmr2)
#
#def nister_err(pt1, pt2, F, K):
#    E = np.einsum('ba,...bc,cd->...ad', K, F, K)

class IntrinsicSolverRANSAC(object):
    def __init__(self, w, h, method='hz'):
        self.w_ = w
        self.h_ = h
        # internal solver handle
        if method == 'hz':
            self.rsc_ = RANSACModel(
                    n_model=3,
                    model_fn=self.rsc_model,
                    err_fn=self.rsc_err,
                    thresh=1e-5, # don't really know what the right thresh is
                    prob=0.999
                    )
            self.solver_ = KruppaSolver(verbose=0)
        elif method == 'mc':
            self.rsc_ = RANSACModel(
                    n_model=5,
                    model_fn=self.rsc_model,
                    err_fn=self.rsc_err,
                    thresh=1e-5, # don't really know what the right thresh is
                    prob=0.999
                    )
            self.solver_ = KruppaSolverMC(verbose=0)
        elif method == 'esv':
            self.rsc_ = RANSACModel(
                    n_model=15, # overparametrized; maybe more stable
                    model_fn=self.rsc_model,
                    err_fn=self.rsc_err,
                    thresh=1e-5, # don't really know what the right thresh is
                    prob=0.999
                    )
            self.solver_ = ESVSolver(w,h,verbose=True)

        # data
        self.cache_ = {}
        self.K_  = None # current K estimate
        self.Fs_ = None # fundamental matrices

    def rsc_model(self, idx):
        K0 = self.K_
        Ws = (self.Ws_[idx] if (self.Ws_ is not None) else None)

        if isinstance(self.solver_, ESVSolver):
            # TODO : maybe at least support Ws input for ESV
            K = self.solver_(self.Fs_[idx], Ws)
        else:
            K = self.solver_(K0, self.Fs_[idx], Ws)
        return K

    def rsc_err(self, model):
        Fs = self.Fs_
        if model is None:
            return np.full(len(self.Fs_), np.inf)

        # Kruppa coefficients matrix
        A = model
        K = A.dot(A.T)

        u1,u2,u3 = [self.cache_[k] for k in ['u1','u2','u3']]
        v1,v2,v3 = [self.cache_[k] for k in ['v1','v2','v3']]
        s1,s2    = [self.cache_[k] for k in ['s1','s2']]

        nmr1 = mul3(v2,K,v2,np=np)
        dmr1 = (s1*s1) * mul3(u1,K,u1,np=np)
        e1   = (nmr1 / dmr1)

        nmr2 = -mul3(v2,K,v1,np=np)
        dmr2 = (s1*s2) * mul3(u1,K,u2,np=np)
        e2   = (nmr2 / dmr2)

        nmr3 = mul3(v1,K,v1,np=np)
        dmr3 = (s2*s2) * mul3(u2,K,u2,np=np)
        e3   = (nmr3 / dmr3)

        err12 = ((e1 - e2)).ravel()
        err23 = ((e2 - e3)).ravel()
        err31 = ((e1 - e3)).ravel()
        err = np.square(err12) + np.square(err23) + np.square(err31)

        #plt.hist(err, bins='auto')
        #plt.show()

        #apply huber loss
        #err = np.where(err < 0.01, err, 2*err**0.5-1)

        if self.Ws_ is not None:
            err = self.Ws_ * err
        print '20-50-80', [np.percentile(err, e) for e in [20,50,80]]
        return err
    
    def __call__(self, A, Fs, Ws=None, max_it=512):
        self.K_ = A
        self.Fs_ = Fs
        self.Ws_ = Ws

        U, S, Vt = np.linalg.svd(Fs)
        u1, u2, u3 = U[...,:,0], U[...,:,1], U[...,:,2]
        v1, v2, v3 = Vt[...,0,:], Vt[...,1,:], Vt[...,2,:]
        s1, s2  = S[...,0], S[...,1]
        for k in ['u1','u2','u3','v1','v2','v3','s1','s2']:
            self.cache_[k] = vars()[k]

        n_it, res = self.rsc_(len(Fs), max_it)
        if res is not None:
            return n_it, res['model'], res['inl']
        return 0, None, np.zeros(len(Fs), dtype=np.bool)
