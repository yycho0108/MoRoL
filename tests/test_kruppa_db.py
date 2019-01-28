from opt.kruppa import KruppaSolver, KruppaSolverMC
from opt.dist import DistSolver
from core.match import Matcher
from utils import cv_wrap as W
from utils import vmath as M
import viz as V
import cv2
import numpy as np

def mcheck(x):
    if x is None:
        return False
    if np.any(np.isnan(x)):
        return False
    return True

def main():
    w, h = 640, 480
    solver0 = KruppaSolverMC()
    solver = KruppaSolver()
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
                    ransacReprojThreshold=4.0,
                    confidence=0.999
                    )

            #print 'rank', np.linalg.matrix_rank(F)
            r_err = np.einsum('...a,ab,...b', M.to_h(pt_b), F, M.to_h(pt_a))
            r_err = np.sqrt(np.square(r_err).mean())
            if r_err < 1.0:
                mim = V.draw_matches(img0, img1, pt_a, pt_b)
                cv2.imshow('mim', mim)
                k = cv2.waitKey(1)
                if k == 27:
                    break
                Fs.append( F )

    K1 = solver0(K0, Fs)
    #if mcheck(K1):
    #    K0 = K1
    print 'K1', K1

    K1 = solver(K0, Fs)
    #if mcheck(K1):
    #    K0 = K1
    print 'K1', K1

if __name__ == "__main__":
    main()
