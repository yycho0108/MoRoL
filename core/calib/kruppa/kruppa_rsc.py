from opt.ransac import RANSACModel
from core.calib.kruppa import KruppaSolver, KruppaSolverMC
import numpy as np
from core.calib.kruppa.common import mul3

class KruppaSolverRANSAC(object):
    def __init__(self):
        self.rsc_ = RANSACModel(
                n_model=3,
                model_fn=self.rsc_model,
                err_fn=self.rsc_err,
                thresh=0.1, # don't really know what the right thresh is
                prob=0.999
                )
        self.cache_ = {}
        self.solver_ = KruppaSolver(verbose=0)
        self.K_  = None # current K estimate
        self.Fs_ = None # fundamental matrices

    def rsc_model(self, idx):
        K0 = self.K_
        K = self.solver_(K0, self.Fs_[idx], self.Ws_[idx])
        return K

    def rsc_err(self, model):
        # assume solver SVD cache was initialized from rsc_model()
        Fs = self.Fs_
        if model is None:
            return np.full(len(self.Fs_), np.inf)

        K = self.solver_.A2K(model)
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

        #err12 = nmr1 * dmr2 - nmr2 * dmr1
        #err23 = nmr2 * dmr3 - nmr3 * dmr2
        #err31 = nmr3 * dmr1 - nmr1 * dmr3

        err12 = ((e1 - e2)).ravel()
        err23 = ((e2 - e3)).ravel()
        err31 = ((e1 - e3)).ravel()
        err = np.square(err12) + np.square(err23) + np.square(err31)
        print 'median err', np.median(err)
        return err
    
    def __call__(self, A, Fs, Ws, max_it=256):
        self.K_ = A
        self.Fs_ = Fs
        self.Ws_ = Ws

        U, S, Vt = np.linalg.svd(Fs)
        u1, u2, u3 = U[...,:,0], U[...,:,1], U[...,:,2]
        v1, v2, v3 = Vt[...,0,:], Vt[...,1,:], Vt[...,2,:]
        s1, s2  = S[...,0], S[...,1]
        for k in ['u1','u2','u3','v1','v2','v3','s1','s2']:
            self.cache_[k] = vars()[k]

        try:
            n_it, res = self.rsc_(len(Fs), max_it)
            if res is not None:
                return n_it, res['model'], res['inl']
        except Exception as e:
            print 'exception', e
            pass
        return 0, None, np.zeros(len(Fs), dtype=np.bool)
