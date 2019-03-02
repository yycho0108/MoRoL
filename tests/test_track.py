import cv2
import numpy as np
from core.track import Tracker
from core.match import Matcher
from utils import vmath as M
import viz as V
from collections import defaultdict
from matplotlib import pyplot as plt

def xyxy2xywh(b):
    x0,y0,x1,y1 = b
    w = x1-x0
    h = y1-y0
    return x0,y0,w,h

def xywh2xyxy(b):
    x0,y0,w,h = b
    x1 = x0+w
    y1 = y0+h
    return x0,y0,x1,y1

class TrackTest(object):
    def __init__(self):
        orb = cv2.ORB_create(
            nfeatures=1024,
            scaleFactor=1.2,
            nlevels=8,
            # NOTE : scoretype here influences response-based filters.
            #scoreType=cv2.ORB_FAST_SCORE,
            scoreType=cv2.ORB_HARRIS_SCORE,
            )
        self.orb_ = orb
        self.des_ = orb
        #self.des_ = cv2.xfeatures2d.BoostDesc_create(
        #        desc = cv2.xfeatures2d.Boost
        #        )
        self.track_ = Tracker()
        self.match_ = Matcher(des=orb)
        self.kcf_   = cv2.TrackerKCF_create()

        cv2.namedWindow('win')
        cv2.setMouseCallback('win', self.mouse_cb)
        #self.cam_ = cv2.VideoCapture(0)
        self.cam_ = cv2.VideoCapture('/home/jamiecho/Downloads/scan_20190212-233625.h264')
        self.cam_.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam_.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.quit_ = False

        self.m0_ = None
        self.m1_ = None
        self.rect_ = None

        # data
        self.img_ = None
        self.prv_ = None
        self.msk_ = None

        self.map_ = {
                'trk' : np.empty(0, dtype=np.bool),
                'kpt' : np.empty((0,2), dtype=np.float32),
                'des' : np.empty((0,32), dtype=np.uint8)
                }
        self.col_ = defaultdict(
                lambda : np.random.randint(0, 255, size=(3,), dtype=int)
                )

    def read(self):
        if self.img_ is not None:
            res, _ = self.cam_.read(self.img_)
        else:
            res, img = self.cam_.read()
            self.img_ = img
        return res
    
    def key_cb(self, key):
        if key in [27, ord('q')]:
            self.quit_ = True

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.m0_ = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.m1_ = (x, y)
        elif event in [cv2.EVENT_LBUTTONUP]:
            (x0,y0), (x1,y1) = self.m0_, self.m1_
            rect = tuple(int(np.round(x)) for x in (
                    min(x0,x1), min(y0,y1),
                    max(x0,x1), max(y0,y1)
                    ))
            self.rect_ = rect

    def step(self):
        img = self.img_
        #kpt, des = self.orb_.detectAndCompute(img, None)
        kpt = self.orb_.detect(img, None)
        kpt, des = self.des_.compute(img, kpt)
        #kpt, des = self.orb_.detectAndCompute(img, None)


        #plt.clf()
        #plt.hist( [k.response for k in kpt] )
        # ~ 0.0005 appears to be reasonable?
        #plt.pause( 0.001 )

        if self.prv_ is None:
            if self.rect_ is not None:
                r = self.rect_
                msk = np.zeros(img.shape[:2], dtype=np.uint8)
                msk[r[1]:r[3], r[0]:r[2]] = 255
                self.msk_ = msk
                #kpt, des = self.orb_.detectAndCompute(img, msk)
                kpt = self.orb_.detect(img, None)
                kpt, des = self.des_.compute(img, kpt)
                self.prv_ = (img.copy(), kpt, des)

                self.kcf_.init(img, xyxy2xywh(r) )
                #msk_roi = msk.copy()
                #self.m_bg_ = np.zeros((1,65), np.float64)
                #self.m_fg_ = np.zeros((1,65), np.float64)
                #cv2.grabCut(img, msk_roi, r,
                #        self.m_bg_, self.m_fg_,
                #        iterCount=5, # itercount
                #        mode=cv2.GC_INIT_WITH_RECT)
                #msk_b = np.logical_or(msk_roi == cv2.GC_FGD, msk_roi == cv2.GC_PR_FGD)
                #cv2.imshow('msk_roi', msk_b.astype(np.uint8) * 255)
                #self.kcf_.init(img * msk_b[..., None], xyxy2xywh(r) )
            return
        
        suc, box = self.kcf_.update(img)
        self.rect_ = xywh2xyxy(box)

        img0, kpt0, des0 = self.prv_
        img1 = self.img_
        kpt0 = np.asarray(kpt0, dtype=cv2.KeyPoint)

        pt0 = cv2.KeyPoint.convert( kpt0 )

        q_idx = np.arange( len(self.map_['trk']) )
        trk = self.map_['trk']
        desl = self.map_['des'][q_idx]
        ptl0 = self.map_['kpt'][q_idx]

        i_trk = np.where(trk)[0]

        i_m_s0, i_m_sl = self.match_(des0, desl,
                **Matcher.PRESET_SOFT
                )

        i_m_h0, i_m_hl = self.match_(des0, desl,
                **Matcher.PRESET_HARD
                )

        # new : "fail" soft match
        # propagates pt0 from current frame
        i_n = M.invert_index(i_m_s0, len(des0))
        ptn0 = pt0[i_n]
        ptn1, m_n = self.track_(img0, img1, ptn0, return_msk=True)
        ptn0 = ptn0[m_n]
        ptn1 = ptn1[m_n]
        # TODO : potentially apply more filters here

        # old : tracked
        # propagates pt0 from track
        pto0 = ptl0[trk]
        pto1, m_o = self.track_(img0, img1, pto0, return_msk=True)

        #cv2.imshow('img0', img0)
        #cv2.imshow('img1', img1)
        #cv2.waitKey(0)

        pto0 = pto0[m_o]
        pto1 = pto1[m_o]

        #o_iol = np.where( trk & m_o ) [0]
        o_iol = i_trk[m_o]

        # recovery : "pass" hard match + untracked
        # propagates pt0 from current frame
        # i_r = [i for (i, ir) in zip(i_m_h0, i_m_hl) if not trk[q_idx][ir] ]
        msk_r = ~trk[q_idx][i_m_hl]
        i_r = i_m_h0[msk_r]
        ptr0 = pt0[i_r]
        ptr1, m_r = self.track_(img0, img1, ptr0, return_msk=True)
        ptr0 = ptr0[m_r]
        ptr1 = ptr1[m_r]

        #print 'recovery'
        #print [k.response for k in kpt0[i_r][m_r]]

        # pt[~rk][q_idx][i_m_hl]
        o_irl = q_idx[i_m_hl[msk_r][m_r]]

        #o_irl = np.intersect1d(M.invert_index(i_trk, len(self.map_['trk'])), q_idx[i_m_hl])
        #o_irl = o_irl[m_r]
        #o_irl = np.setdiff1d(q_idx[i_m_hl][msk_r][m_r], i_trk)

        # suppress : pass soft match + tracked
        # (no_op)

        # update tracking flags
        self.map_['trk'][i_trk[~m_o]] = False
        self.map_['trk'][o_irl] = True

        # update tracking points
        self.map_['kpt'][o_irl] = ptr1
        self.map_['kpt'][o_iol] = pto1

        # insert points
        if self.map_['kpt'].size == 0:
            self.map_['kpt'] = np.concatenate([
                self.map_['kpt'], ptn1])
            self.map_['des'] = np.concatenate([
                self.map_['des'], des0[i_n][m_n]])
            self.map_['trk'] = np.concatenate([
                self.map_['trk'], np.ones(len(ptn1), dtype=np.bool)])

        # update prior
        self.prv_ = (self.img_.copy(), kpt, des)

        return (ptn0, ptn1), (pto0, pto1, o_iol), (ptr0, ptr1, o_irl)

    def __call__(self):
        first_viz = True
        while not self.quit_:
            if not self.read():
                break
            self.step()
            cv2.imshow('win', self.img_ )

            # show tracking points
            viz = self.img_.copy()
            trk_i = np.where(self.map_['trk'])[0]
            print 'tracking : ', len(trk_i)
            if self.rect_ is not None:
                cv2.rectangle(viz, 
                        tuple(map(int, self.rect_[:2])),
                        tuple(map(int, self.rect_[2:])),
                        #tuple(map(int, self.rect_)),
                        color=(0,0,255),
                        thickness=1
                        )
            for i, pt in zip(trk_i, self.map_['kpt'][trk_i]):
                p = tuple(int(np.round(e)) for e in pt)
                c = self.col_[i]
                cv2.circle(viz,
                            center=p,
                            radius=3,
                            color=c,
                            thickness = 1)
            if first_viz and len(trk_i) > 0:
                first_viz = False
                cv2.imshow('reference', viz)
            cv2.imshow('viz', viz)


            k = cv2.waitKey(0)
            self.key_cb( k )


def main():
    TrackTest()()

if __name__ == "__main__":
    main()
