import numpy as np
import cv2

from utils import cv_wrap as W

class EpiTracker(object):
    """
    Robust epipolar tracking
    """
    def __init__(self, feat, track, match):
        self.feat_   = feat
        self.track_ = track
        #self.match_ = match

    def __call__(self,
            img_a, img_b,
            pt_a, pt_b=None
            ):
        kp_a, des_a = self.feat_(img_a)
        kp_b, des_b = self.feat_(img_b)
        ##m_a, m_b, s = self.match_(des_a, des_b, return_score=True)

        ## sort match
        #n_pt = 16 # use 16 points
        #m_a = m_a[ np.argsort(s) ]
        #m_b = m_b[ np.argsort(s) ]

        # initialize fundamental matrix
        #F, _ = W.F(p1[m_a[-16:]], p2[m_b[-16:]], cameraMatrix=K, **self.pEM_)
        F, _ = W.F(p1, p2, cameraMatrix=K, **self.pEM_)

        # compute epipoles from null-space of F
        ep2, s, ep1T = np.linalg.svd(F)
        i = np.argmin(s)
        ep1 = M.from_h(ep1T[i,:])
        ep2 = M.from_h(ep2[:,i])

        # compute epilines
        el1 = F.T.dot( M.to_h(ep2)) # epiline 2->1
        el2 = F.dot( M.to_h(ep1)) # epiline 1->2
