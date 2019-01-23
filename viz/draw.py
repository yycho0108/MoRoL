import numpy as np
import cv2
from utils import vmath as M

def draw_lines(img1,img2,lines,pts1,pts2,cols,
        draw_pt=False
        ):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    h,w = img1.shape[:2]
    for r,pt1,pt2,color in zip(lines,pts1,pts2,cols):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [w, -(r[2]+r[0]*w)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        if draw_pt:
            img1 = cv2.circle(img1, tuple(M.rint(pt1)),5,color,-1)
            img2 = cv2.circle(img2, tuple(M.rint(pt2)),5,color,-1)
    return img1,img2

def draw_matches(img1, img2, pt1, pt2,
        msk=None,
        radius=3
        ):
    h,w = np.shape(img1)[:2]
    pt1 = np.round(pt1).astype(np.int32)
    pt2 = np.round(pt2 + [[w,0]]).astype(np.int32)

    mim = np.concatenate([img1, img2], axis=1)
    mim[:,w ] = 0
    mim0 = mim.copy()

    if msk is None:
        msk = np.ones(len(pt1), dtype=np.bool)

    n = msk.sum()
    col = np.random.randint(255, size=(n,3))

    for (p1, p2, c) in zip(pt1[msk], pt2[msk], col):
        p1 = tuple(p1)
        p2 = tuple(p2)
        cv2.line(mim, p1, p2, c, 2)
    mim = cv2.addWeighted(mim0, 0.5, mim, 0.5, 0.0)

    for (p1, p2, c) in zip(pt1[msk], pt2[msk], col):
        cv2.circle(mim, tuple(p1), radius, c, 2)
        cv2.circle(mim, tuple(p2), radius, c, 2)

    for p in pt1[~msk]:
        cv2.circle(mim, tuple(p), radius, (255,0,0), 1)

    for p in pt2[~msk]:
        cv2.circle(mim, tuple(p), radius, (255,0,0), 1)
    return mim

def draw_epilines(img1, img2, pt1, pt2,
        F=None,
        E=None,
        K=None,
        sample=32,
        copy=True
        ):

    # draw a fraction of the epilines
    if sample > 0:
        idx = np.random.choice(len(pt1), size=sample)
    else:
        idx = np.arange(len(pt1))

    # avoid modifying img1/img2
    if copy:
        img1 = img1.copy()
        img2 = img2.copy()

    if F is None:
        F = M.E2F(E, K=K)

    pt1, pt2 = pt1[idx], pt2[idx]

    lines1 = cv2.computeCorrespondEpilines(pt2.reshape(-1,1,2), 2, F)\
            .reshape(-1,3)
    lines2 = cv2.computeCorrespondEpilines(pt1.reshape(-1,1,2), 1, F)\
            .reshape(-1,3)

    cols = [tuple(np.random.randint(0,255,3).tolist()) for _ in idx]

    draw_lines(img1, img2, lines1, pt1, pt2, cols, draw_pt=True)
    draw_lines(img2, img1, lines2, pt2, pt1, cols, draw_pt=True)

    return img1, img2
