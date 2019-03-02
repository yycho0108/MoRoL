from core.calib.focal import FocalSolverStrum
from core.calib.esv import ESVSolver
from core.calib.intrinsic_rsc import IntrinsicSolverRANSAC

from opt.dist import DistSolver
from core.match import Matcher
from core.track import Tracker
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

def detect_and_compute(img, det, des):
    k = det.detect(img, None)
    #rsp = [e.response for e in k]
    #k = np.asarray(k, dtype=cv2.KeyPoint)
    #k = k[np.argsort(rsp)[::-1]]
    #k = k[:256]
    #k = SSC(k, 256, 0.1, img.shape[1], img.shape[0])
    k, d = des.compute(img, k)
    return k, d

def main():
    #feat = cv2.ORB_create(nfeatures=1024)
    #feat = cv2.AKAZE_create()
    #feat = cv2.xfeatures2d.SIFT_create(512)

    det = cv2.FastFeatureDetector_create(40, True)
    des = cv2.xfeatures2d.BoostDesc_create()

    matcher = Matcher(des=des)
    match_cfg = Matcher.PRESET_HARD.copy()
    tracker = Tracker()
    match_cfg['maxd'] = 64.0

    imgs = np.load('/tmp/db_imgs.npy')
    #kpts = None
    #dess = None
    kpts = np.load('/tmp/db_kpts.npy')
    dess = np.load('/tmp/db_dess.npy')

    #src = np.setxor1d(range(15), [8, 10])
    src = range(15)
    #imgs = [cv2.imread('/home/jamiecho/Downloads/images/%03d.tif' % i) for i in src]
    #imgs = [e for e in imgs if e is not None]
    #imgs = [cv2.imread('/home/jamiecho/Downloads/images/%03d.tif' % i) for i in [2,3,4,5,6,9]]

    if kpts is None:
        print 'building kpt/des cache ...'
        kpts = []
        dess = []
        for img in imgs:
            k, d = detect_and_compute(img, det, des)
            kpts.append(np.asarray(cv2.KeyPoint.convert(k)))
            dess.append( d )
        print 'kpt/des cache complete!'

    Fs = []
    Ws = []
    for i in range(0, len(imgs)):
        print '{}/{}'.format(i, len(imgs))
        for j in range(i+1, i+4):
        #for j in range(i-8, i+8):
        #for j in range(i+1, len(imgs)):
            if i==j: continue
            if j<0 or j>=len(imgs): continue

            # use db input
            img0, kpt0, des0 = [e[i] for e in [imgs,kpts,dess]]
            img1, kpt1, des1 = [e[j] for e in [imgs,kpts,dess]]

            if des0 is None or des1 is None:
                continue
            if len(des0) <= 2 or len(des1) <= 2:
                continue

            h, w = img0.shape[:2]

            kpt0_1, idx_t= tracker(img0, img1, kpt0)
            pt_a = kpt0[idx_t]
            pt_b = kpt0_1[idx_t]
            if len(pt_a) <= 16:
                continue

            #i_m_h0, i_m_h1 = matcher(des0, des1,
            #        **match_cfg
            #        )
            #if len(i_m_h0) < 32:
            #    continue
            #pt_a = kpt0[i_m_h0]
            #pt_b = kpt1[i_m_h1]

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
            #if True:
                #mim = V.draw_matches(img0, img1, pt_a, pt_b,
                #        msk=msk
                #        )
                #cv2.imshow('mim', mim)
                #k = cv2.waitKey(1)
                #if k == ord('q'):
                #    break
                #if k == 27:
                #    return
                Fs.append( F )
                Ws.append( n_in )
                if len(Fs) > 0 and len(Fs) % 10 == 0:
                    print len(Fs)

    print 'fs-len', len(Fs)

    Fs = np.asarray(Fs)
    Ws = np.asarray(Ws, dtype=np.float32)
    Ws /= Ws.max()

    #Fs = Fs[np.random.choice(len(Fs), size=128)]
    #Fs = Fs[np.random.choice(len(Fs), size=128)]

    # solve focal length first
    #f = foc_solver(Fs)
    #K0[0,0] = K0[1,1] = f
    #print('updated Kmat through focal length initialization : {}'.format(K0))

    #K1 = solver0(K0, Fs)
    ##if mcheck(K1):
    ##    K0 = K1
    #print 'K1', K1

    solver = ESVSolver(w, h)
    K1 = solver(Fs, Ws)
    
    #K0 = np.float32([
    #    (w+h), 0.0, w/2.0,
    #    0.0, (w+h), h/2.0,
    #    0, 0, 1]).reshape(3,3)
    #solver = IntrinsicSolverRANSAC(w,h,method='esv')
    #K1 = solver(K0, Fs, Ws)

    #print 'best?'
    #print solver.solver_.best_

    #if mcheck(K1):
    #    K0 = K1
    print 'K1', K1

if __name__ == "__main__":
    main()
