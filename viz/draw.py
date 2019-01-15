import numpy as np
import cv2

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

