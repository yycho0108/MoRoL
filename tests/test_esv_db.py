from core.calib.focal import FocalSolverStrum
from core.calib.esv import ESVSolver

from opt.dist import DistSolver
from core.match import Matcher
from utils import cv_wrap as W
from utils import vmath as M
import viz as V
import cv2
import numpy as np

import autograd.numpy as anp
from autograd import jacobian

def mcheck(x):
    if x is None:
        return False
    if np.any(np.isnan(x)):
        return False
    return True

class FundCov(object):
    def __init__(self):
        self.pt_a_ = None
        self.pt_b_ = None
        self.jac = jacobian(self.err_anp)

    def err_anp(self, F):
        return self.err(F, np=anp)

    def err(self, F, np=np):
        pt_a = self.pt_a_
        pt_b = self.pt_b_
        e = np.einsum('...a,ab,...b', M.to_h(pt_b), F.reshape(3,3), M.to_h(pt_a))
        return np.ravel(e)

    @staticmethod
    def jac_to_cov(J):
        """ from scipy/optimize/minpack.py#L739 """
        _, s, VT = np.linalg.svd(J, full_matrices=False)
        thresh = np.finfo(np.float32).eps * max(J.shape) * s[0]
        s = s[s > thresh]
        VT = VT[:s.size]
        cov = np.dot(VT.T / s**2, VT)
        return cov

    def __call__(self, F, pt_a, pt_b):
        self.pt_a_ = pt_a
        self.pt_b_ = pt_b
        J = self.jac(F.ravel())
        return self.jac_to_cov(J)

def jac_F(F, pt_a, pt_b, np=np):
    jacobian(err_F,self.err_anp)#, np=anp)
    return 

def main():
    w, h = 640, 480
    solver = ESVSolver(w,h)
    K0 = np.float32([
        (w+h), 0.0, w/2.0,
        0.0, (w+h), h/2.0,
        0, 0, 1]).reshape(3,3)
    #K0 = np.float32([200,0,200,0,200,200,0,0,1]).reshape(3,3)
    #K0 = np.float32([
    #    1260,0,280,0,1260,230,0,0,1]).reshape(3,3)
    feat = cv2.ORB_create(nfeatures=1024)
    matcher = Matcher(des=feat)

    imgs = np.load('/tmp/db_imgs.npy')
    kpts = np.load('/tmp/db_kpts.npy')
    dess = np.load('/tmp/db_dess.npy')
    Fs = []
    Ws = []

    for i in range(0, len(imgs)):
        print '{}/{}'.format(i, len(imgs))
        for j in range(i-8, i+8):
            if i==j: continue
            if j<0 or j>=len(imgs): continue

            img0, kpt0, des0 = [e[i] for e in [imgs,kpts,dess]]
            img1, kpt1, des1 = [e[j] for e in [imgs,kpts,dess]]

            i_m_h0, i_m_h1 = matcher(des0, des1,
                    **Matcher.PRESET_HARD
                    )

            if len(i_m_h0) < 32:
                continue

            pt_a = kpt0[i_m_h0]
            pt_b = kpt1[i_m_h1]

            F, msk = W.F(pt_a, pt_b,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=2.0,
                    confidence=0.999
                    )
            #print 'rank', np.linalg.matrix_rank(F)
            n_in = np.count_nonzero(msk)
            r_in = n_in / float(msk.size)
            #print 'n_in / r_in : {}, {}'.format(n_in, r_in)

            r_err = np.einsum('...a,ab,...b', M.to_h(pt_b), F, M.to_h(pt_a))
            r_err = np.sqrt(np.square(r_err).mean())
            if n_in > 64 and r_in > 0.8 and r_err < 1.0:
                #mim = V.draw_matches(img0, img1, pt_a, pt_b)
                #cv2.imshow('mim', mim)
                #k = cv2.waitKey(1)
                #if k == ord('q'):
                #    break
                #if k == 27:
                #    return
                Fs.append( F )

    print 'fs-len', len(Fs)

    Fs = np.asarray(Fs)
    Ws = np.asarray(Ws)
    #Ws *= Ws.size / Ws.sum()

    Fs = Fs[np.random.choice(len(Fs), size=128)]
    #Fs = Fs[np.random.choice(len(Fs), size=128)]

    # solve focal length first
    #f = foc_solver(Fs)
    #K0[0,0] = K0[1,1] = f
    #print('updated Kmat through focal length initialization : {}'.format(K0))


    #K1 = solver0(K0, Fs)
    ##if mcheck(K1):
    ##    K0 = K1
    #print 'K1', K1

    K1 = solver(Fs)
    #if mcheck(K1):
    #    K0 = K1
    print 'K1', K1

if __name__ == "__main__":
    main()
