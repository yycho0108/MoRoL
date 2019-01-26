from opt.dist import DistSolver
from core.match import Matcher
import cv2
import numpy as np

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

        img = cv2.drawKeypoints(img, kpt, img)
        cv2.imshow('win', img)

        kpt0 = kpt1
        des0 = des1

        k = cv2.waitKey( 1 )
        if k == 27:
            break

if __name__ == "__main__":
    main()
