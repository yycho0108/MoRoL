import numpy as np
from autograd import numpy as anp
from autograd import jacobian
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
from utils import vmath as M

class ESVSolver(object):
    def __init__(self, w, h, verbose=True):
        self.w_, self.h_ = w,h
        self.Fs_ = None
        self.Ws_ = None
        self.verbose_ = verbose
        self.cfg_ = {}
        self.params_ = dict(
            ftol=1e-6,
            xtol=1e-6,
            #gtol=1e-16,
            loss='linear',
            max_nfev=1024,
            method='trf',
            #method='lm',
            #method='lm',
            verbose=(2 if verbose else 0),
            #tr_solver='lsmr',
            tr_solver='exact',
            f_scale=0.1
            #f_scale=1.0
            #x_scale='jac',
            )
        self.jac0 = jacobian(lambda x : self.cost0(x, np=anp))
        self.jac1 = jacobian(lambda x : self.cost1(x, np=anp))

    def data(self, params):
        # arguments should be supplied either through unraveled params or cfg
        cfg = self.cfg_
        res = {}
        for k,(s,r) in cfg['param'].items():
            res[k] = params[r].reshape(s)
        for k,v in cfg['const'].items():
            res[k] = v
        return res

    @staticmethod
    def trace(x, np=np):
        # hack because "decent" methods do not work
        return np.diagonal(x, axis1=-1, axis2=-2).sum(axis=-1)
        #return np.trace(x, axis1=-2, axis2=-1)
        #return np.einsum('...ii', x)

    def cost0(self, params, np=np):
        data = self.data(params)
        fx,fy,cx,cy,sk = [data[e] for e in ['fx','fy','cx','cy','sk']]
        K = np.array([fx,sk,cx,0,fy,cy,0,0,1]).reshape(3,3)
        Fs = data['Fs']

        E = np.einsum('ab,...bc,cd->...ad', K.T,Fs,K)
        ET = E.swapaxes(-2,-1)
        EET = np.einsum('...ac,...cb->...ab', E, ET)

        EETEET = np.einsum('...ab,...bc->...ac', EET, EET)

        #dmr = np.square(self.trace(EET,np=np))
        #nmr = dmr - 2.0 * self.trace(EETEET,np=np)
        #err = nmr / dmr

        err = 1.0 - 2.0 * self.trace(EETEET,np=np) / np.square(self.trace(EET,np=np))
        #plt.hist(err, bins='auto')
        #plt.show()
        if self.Ws_ is None:
            return err.ravel()
        return self.Ws_ * err.ravel()

    def cost1(self, params, np=np):
        data = self.data(params)
        fx,fy,cx,cy,sk = [data[e] for e in ['fx','fy','cx','cy','sk']]
        K = np.array([fx,sk,cx,0,fy,cy,0,0,1]).reshape(3,3)
        Fs = data['Fs']
        Es = np.einsum('ab,...bc,cd->...ad', K.T,Fs,K)
        s = np.linalg.svd(Es, full_matrices=False, compute_uv=False)
        s1 = s[..., 0]
        s2 = s[..., 1]
        err = (s1 - s2) / (s1 + s2)
        if self.Ws_ is None:
            return err.ravel()
        return self.Ws_ * err.ravel()
    
    def solve(self, params, err, jac):
        return least_squares(err, params,
                jac=jac,
                **self.params_
                )

    def step0(self):
        # TODO : make this step closed-form analytical
        # calibrate focal length assuming unit aspect
        # could technically use FocalSolverStrum()
        # param : fx
        # cost : cost0(4.10)
        param = self.K_[0,0]
        self.cfg_['param'] = {
                'fx' : ([], np.r_[0]),
                'fy' : ([], np.r_[0])
                }
        self.cfg_['const'] = {
                'cx' : self.K_[0,2],
                'cy' : self.K_[1,2],
                'sk' : self.K_[0,1], # skew
                'Fs' : self.Fs_
                }
        res = self.solve(param, self.cost0, self.jac0)
        data = self.data(res.x)
        fx,fy,cx,cy,sk = [data[e] for e in ['fx','fy','cx','cy','sk']]
        fx, fy = [np.abs(e) for e in [fx,fy]]
        K = np.array([fx,sk,cx,0,fy,cy,0,0,1]).reshape(3,3)
        return K

    def step1(self):
        # estimate aspect ratio
        # param : fx, fy
        # cost : cost0(4.10)
        param = self.K_[(0,1),(0,1)]
        self.cfg_['param'] = {
                'fx' : ([], np.r_[0]),
                'fy' : ([], np.r_[1])
                }
        self.cfg_['const'] = {
                'cx' : self.K_[0,2],
                'cy' : self.K_[1,2],
                'sk' : self.K_[0,1], # skew
                'Fs' : self.Fs_
                }
        res = self.solve(param, self.cost0, self.jac0)
        data = self.data(res.x)
        fx,fy,cx,cy,sk = [data[e] for e in ['fx','fy','cx','cy','sk']]
        K = np.array([fx,sk,cx,0,fy,cy,0,0,1]).reshape(3,3)
        return K

    def step2(self):
        # estimate principal point
        # param : cx, cy
        # cost : cost1(4.13)
        param = self.K_[(0,1),(2,2)]
        self.cfg_['param'] = {
                'cx' : ([], np.r_[0]),
                'cy' : ([], np.r_[1])
                }
        self.cfg_['const'] = {
                'fx' : self.K_[0,0],
                'fy' : self.K_[1,1],
                'sk' : self.K_[0,1], # skew
                'Fs' : self.Fs_
                }
        res = self.solve(param, self.cost1, self.jac1)
        data = self.data(res.x)
        fx,fy,cx,cy,sk = [data[e] for e in ['fx','fy','cx','cy','sk']]
        K = np.array([fx,sk,cx,0,fy,cy,0,0,1]).reshape(3,3)
        return K

    def step3(self):
        # estimate focal lengths
        # param : fx, fy
        # cost : cost0(4.10)
        param = self.K_[(0,1),(0,1)]
        self.cfg_['param'] = {
                'fx' : ([], np.r_[0]),
                'fy' : ([], np.r_[1])
                }
        self.cfg_['const'] = {
                'cx' : self.K_[0,2],
                'cy' : self.K_[1,2],
                'sk' : self.K_[0,1], # skew
                'Fs' : self.Fs_
                }
        res = self.solve(param, self.cost0, self.jac0)
        data = self.data(res.x)
        fx,fy,cx,cy,sk = [data[e] for e in ['fx','fy','cx','cy','sk']]
        K = np.array([fx,sk,cx,0,fy,cy,0,0,1]).reshape(3,3)
        return K

    def step4(self):
        # refinement of all parameters
        # param : fx, fy, cx, cy,(sk)
        # cost : cost1(4.13)
        param = self.K_[(0,1,0,1),(0,1,2,2)]
        self.cfg_['param'] = {
                'fx' : ([], np.r_[0]),
                'fy' : ([], np.r_[1]),
                'cx' : ([], np.r_[2]),
                'cy' : ([], np.r_[3])
                }
        self.cfg_['const'] = {
                'sk' : self.K_[0,1], # skew
                'Fs' : self.Fs_
                }
        res = self.solve(param, self.cost1, self.jac1)
        data = self.data(res.x)
        fx,fy,cx,cy,sk = [data[e] for e in ['fx','fy','cx','cy','sk']]
        K = np.array([fx,sk,cx,0,fy,cy,0,0,1]).reshape(3,3)
        return K

    def __call__(self, Fs, Ws=None):
        self.Fs_ = Fs
        self.Ws_ = Ws

        # initial intrinsic parameters guess
        if self.verbose_:
            print '0 (guess)'
        f0 = (self.w_ + self.h_) / 2.0
        self.K_ = np.float32([
            f0, 0, self.w_/2.0,
            0, f0, self.h_/2.0,
            0,0,1]).reshape(3,3)
        if self.verbose_:
            print self.K_
            print '==============='

        if self.verbose_:
            print '1 (focal)'
        self.K_ = self.step0()
        if self.verbose_:
            print self.K_
            print '==============='

        if self.verbose_:
            print '2 (aspect)'
        self.K_ = self.step1()
        if self.verbose_:
            print self.K_
            print '==============='

        if self.verbose_:
            print '3 (principal)'
        self.K_ = self.step2()
        if self.verbose_:
            print self.K_
            print '==============='

        if self.verbose_:
            print '4 (focal)'
        self.K_ = self.step3()
        if self.verbose_:
            print self.K_
            print '==============='

        if self.verbose_:
            print '5 (all)'
        self.K_ = self.step4()
        if self.verbose_:
            print self.K_
            print '==============='

        if self.verbose_:
            x = self.K_[(0,1,0,1),(0,1,2,2)]
            e0 = self.cost0(x)
            J0 = self.jac0(x)
            e1 = self.cost1(x)
            J1 = self.jac1(x)
            cov0 = M.jac_to_cov(J0, e0)
            std0 = np.sqrt(np.diag(cov0))
            cov1 = M.jac_to_cov(J1, e1)
            std1 = np.sqrt(np.diag(cov1))

            print 'evaluation'
            print cov0
            print 'fx,fy,cx,cy std [0]', std0
            print cov1
            print 'fx,fy,cx,cy std [1]', std1

        return self.K_

def main():
    solver = ESVSolver()

if __name__ == "__main__":
    main()
