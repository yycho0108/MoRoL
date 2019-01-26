from opt.kruppa import KruppaSolver, KruppaSolverMC
from core.match import Matcher
from utils import cv_wrap as W
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
    feat = cv2.ORB_create(nfeatures=1024)
    matcher = Matcher(des=feat)

    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    kpt0 = None
    des0 = None
    Fs = []

    while True:
        res, img = cam.read()
        if not res:
            break

        kpt, des = feat.detectAndCompute(img, None)
        pt = cv2.KeyPoint.convert( kpt )

        if des0 is None:
            kpt0 = pt
            des0 = des
            continue

        kpt1 = pt
        des1 = des

        i_m_h0, i_m_h1 = matcher(des0, des1,
                **Matcher.PRESET_HARD
                )
        pt_a = kpt0[i_m_h0]
        pt_b = kpt1[i_m_h1]

        try:
            F, msk = W.F(pt_a, pt_b,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=0.999,
                    confidence=1.0
                    )
            w = float(np.count_nonzero(msk)) / msk.size
            print('w', w)
            if w > 0.5:
                Fs.append( F )
        except Exception as e:
            print 'exception', e
            continue

        if len(Fs) > 64:
            K1 = solver0(K0, Fs)
            if mcheck(K1):
                K0 = K1
            #K1 = solver(K0, Fs)
            #if mcheck(K1):
            #    K0 = K1
            print('K', K0)
            Fs = []
            kpt0 = None
            des0 = None
        else:
            kpt0 = kpt1
            des0 = des1

        img = cv2.drawKeypoints(img, kpt, img)
        cv2.imshow('win', img)

        k = cv2.waitKey( 1 )
        if k == 27:
            break

if __name__ == "__main__":
    main()