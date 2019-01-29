from core.calib.kruppa import KruppaSolver, KruppaSolverMC
from opt.dist import DistSolver
from core.match import Matcher
from utils import cv_wrap as W
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

def err_F(F, pt_a, pt_b):
    e = np.einsum('...a,ab,...b', pt_a, F, pt_b)
    return e

def main():
    w, h = 640, 480
    dsolver = DistSolver(w, h)
    dsolver.l_ = 8.0
    solver0 = KruppaSolverMC()
    solver = KruppaSolver()
    #K0 = np.float32([
    #    (w+h), 0.0, w/2.0,
    #    0.0, (w+h), h/2.0,
    #    0, 0, 1]).reshape(3,3)
    #K0 = np.float32([200,0,200,0,200,200,0,0,1]).reshape(3,3)
    K0 = np.float32([
        1260,0,280,0,1260,230,0,0,1]).reshape(3,3)
    feat = cv2.ORB_create(nfeatures=1024)
    matcher = Matcher(des=feat)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    db = []

    img0 = None
    kpt0 = None
    des0 = None
    Fs = []

    cv2.namedWindow('win')

    while True:
        res, img = cam.read()
        if not res:
            break

        # initial processing
        kpt, des = feat.detectAndCompute(img, None)
        kpt = cv2.KeyPoint.convert( kpt )
        #kpt = dsolver.undistort(kpt)
        db.append([img, kpt, des])
        if len(db) <= 1:
            continue
        img0, kpt0, des0 = db[-2]
        img1, kpt1, des1 = db[-1]

        i_m_h0, i_m_h1 = matcher(des0, des1,
                **Matcher.PRESET_HARD
                )
        pt_a = kpt0[i_m_h0]
        pt_b = kpt1[i_m_h1]

        try:
            if len(pt_a) > 64:
                F, msk = W.F(pt_a, pt_b,
                        method=cv2.FM_RANSAC,
                        ransacReprojThreshold=2.0,
                        confidence=0.999
                        )
                w = float(np.count_nonzero(msk)) / msk.size
                print('w', w)
                print np.linalg.matrix_rank(F)
                if w > 0.4:
                    c = cov_F(F, pt_a[msk], pt_b[msk])
                    Fs.append( F )
        except Exception as e:
            print 'exception', e
            continue

        print '{}/{}'.format( len(Fs), 512)

        if len(Fs) > 256:
            # save data for reference
            imgs, kpts, dess = zip(*db)
            np.save('/tmp/db_imgs.npy', imgs)
            np.save('/tmp/db_kpts.npy', kpts)
            np.save('/tmp/db_dess.npy', dess)

            K1 = solver0(K0, Fs)
            #if mcheck(K1):
            #    K0 = K1
            print 'K1', K1

            K1 = solver(K0, Fs)
            if mcheck(K1):
                K0 = K1
            print 'K1', K1

            print('K', K0)
            Fs = []
            kpt0 = None
            des0 = None
        else:
            img0 = img1
            kpt0 = kpt1
            des0 = des1

        #img = cv2.drawKeypoints(img, kpt, img)
        if kpt0 is not None and kpt1 is not None:
            mim = V.draw_matches(img0, img1, kpt0, kpt1)
        cv2.moveWindow('win', 500, 500)
        cv2.imshow('win', mim)

        k = cv2.waitKey( 1 )
        if k == 27:
            break

if __name__ == "__main__":
    main()
