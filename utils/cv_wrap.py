import cv2
import numpy as np
from utils import vmath as M

def H(*args, **kwargs):
    h, msk = cv2.findHomography(
            *args, **kwargs)
    msk = msk[:,0].astype(np.bool)
    return h, msk

def E(*args, **kwargs):
    e, msk = cv2.findEssentialMat(
            *args, **kwargs)
    msk = msk[:,0].astype(np.bool)
    return e, msk

def F(*args, **kwargs):
    f, msk = cv2.findFundamentalMat(
            *args, **kwargs)
    msk = msk[:,0].astype(np.bool)
    return f, msk

def project_points(*args, **kwargs):
    pt2, jac = cv2.projectPoints(*args, **kwargs)
    return pt2[:, 0]

def correct_matches(F, pta, ptb):
    pta_f, ptb_f = cv2.correctMatches(F, pta[None,...], ptb[None,...])
    pta_f = pta_f[0]
    ptb_f = ptb_f[0]
    return pta_f, ptb_f

def triangulate_points(Pa, Pb, pta, ptb,
        *args, **kwargs):
    pt_h = cv2.triangulatePoints(
            Pa, Pb,
            pta[None,...],
            ptb[None,...])
    return M.from_h(pt_h.T)
