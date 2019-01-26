from opt.dist import DistSolver
from core.match import Matcher
import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    w, h = 640, 480
    solver = DistSolver(w,h)
    #solver.l_ = 7.0
    feat = cv2.ORB_create(nfeatures=1024)
    matcher = Matcher(des=feat)

    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    kpt0 = None
    des0 = None

    ls = []

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
        l = solver(pt_a, pt_b)
        if l is not None:
            print 'l', l[1]
            print '{}/{}'.format(l[2].sum(), len(pt_a))
            ls.append( l[1] )

        img = cv2.drawKeypoints(img, kpt, img)
        cv2.imshow('win', img)

        kpt0 = kpt1
        des0 = des1

        k = cv2.waitKey( 1 )
        if k == 27:
            break

        if len(ls) > 0:
            ls_lo = np.percentile(ls, 20.0)
            ls_hi = np.percentile(ls, 80.0)
            ls_mid = np.array(ls)[np.logical_and(ls_lo <= ls, ls <= ls_hi)]
            plt.clf()
            plt.hist(ls_mid)
            plt.pause(0.001)

if __name__ == "__main__":
    main()
