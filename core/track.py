class Tracker(object):
    def __init__(self, pLK):
        self.pLK_ = pLK

    def __call__(self, 
            img1, img2,
            pt1, pt2=None,
            thresh=2.0,
            ):
        if pt1.size <= 0:
            # soft fail
            pt2 = np.empty([0,2], dtype=np.float32)
            idx = np.empty([0], dtype=np.int32)
            return pt2, idx

        # stat img
        h, w = np.shape(img2)[:2]

        # copy LK Params
        pLK = self.pLK_.copy()

        # convert to grayscale
        # TODO : check if already gray/mono
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # forward flow
        if pt2 is not None:
            # set initial flow flags
            pLK['flags'] |= cv2.OPTFLOW_USE_INITIAL_FLOW
        pt2, st, err = cv2.calcOpticalFlowPyrLK(
                img1_gray, img2_gray, pt1, pt2,
                **pLK)

        # backward flow
        # unset initial flow flags
        pLK['flags'] &= ~cv2.OPTFLOW_USE_INITIAL_FLOW
        pt1_r, st, err = cv2.calcOpticalFlowPyrLK(
                img2_gray, img1_gray, pt2, None,
                **pLK)

        # override error with reprojection error
        # (default error doesn't make much sense anyways)
        err = np.linalg.norm(pt1 - pt1_r, axis=-1)

        # apply mask
        idx = np.arange(len(pt1))
        msk_in = np.all(np.logical_and(
                np.greater_equal(pt2, [0,0]),
                np.less(pt2, [w,h])), axis=-1)
        msk_st = st[:,0].astype(np.bool)

        # track reprojection error
        msk_err = (err < thresh)
        msk = np.logical_and.reduce([
            msk_err,
            msk_in,
            msk_st
            ])
        idx = idx[msk]

        return pt2, idx

