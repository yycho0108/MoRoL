import numpy as np
from autograd import numpy as anp
from autograd import jacobian
from scipy.optimize import least_squares

class KruppaSolverMC(object):
    """
    http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FUSIELLO3/node4.html
    ftp://mi.eng.cam.ac.uk/pub/reports/mendonca_self-calibration.pdf
    """
    def __init__(self, verbose=2):
        self.params_ = dict(
            ftol=1e-5,
            xtol=1e-7,
            loss='linear',
            #x_scale='jac',
            max_nfev=1024,
            method='lm',
            #method='trf',
            verbose=verbose,
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

    def err_anp(self, params):
        e = self.err(params, np=anp)
        return e

    def err(self, params, np=np):
        Fs = self.Fs_
        A = self.unwrap_A(params, np=np)
        Es = np.einsum('ba,...bc,cd->...ad', A, Fs, A)
        s = np.linalg.svd(Es,full_matrices=False, compute_uv=False)
        #c = (s[..., 0] / s[...,1]) - 1.0
        c = (s[...,0] - s[...,1]) / (s[...,0] + s[...,1])
        return c

    def __call__(self, A, Fs, Ws):
        self.Fs_ = Fs
        self.Ws_ = Ws
        res = least_squares(
                self.err, self.wrap_A(A),
                jac=self.jac,
                x_scale='jac',
                **self.params_)
        A = self.unwrap_A(res.x)
        #print 'K (optimized)'
        #print A
        return A

