from utils.conversions import Conversions
from core.track import Tracker
from core.vgraph import VGraph
from core.vmap import VMap
from utils import vmath as M

import viz as V

class VO(object):
    def __init__(self, cinfo):
        # camera intrinsic / extrinsic parameters
        self.K_ = cinfo['K']
        self.Ki_ = M.inv(K_)
        self.D_ = cinfo['K']
        self.T_c2b_ = cinfo['T']
        self.T_b2c_ = M.Ti(self.T_c2b_)

        # feature detection / description
        orb = cv2.ORB_create(
                nfeatures=1024,
                scaleFactor=1.2,
                nlevels=8,
                # NOTE : scoretype here influences response-based filters.
                scoreType=cv2.ORB_FAST_SCORE,
                #scoreType=cv2.ORB_HARRIS_SCORE,
                )
        det = orb
        des = orb


        # system parameters
        self.pEM_ = dict(
                method=cv2.FM_RANSAC,
                prob=0.999,
                threshold=1.0)
        self.pLK_ = dict(
                winSize = (12,6),
                maxLevel = 4, # == effective winsize up to 32*(2**4) = 512x256
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.003),
                flags = 0,
                minEigThreshold = 1e-3 # TODO : disable eig?
                )
        self.pBA_ = dict(
                ftol=1e-4,
                xtol=np.finfo(float).eps,
                loss='huber',
                max_nfev=1024,
                method='trf',
                verbose=2,
                tr_solver='lsmr',
                f_scale=1.0
                )
        self.pPNP_ = dict(
                iterationsCount=10000,
                reprojectionError=1.0,
                confidence=0.999,
                flags = cv2.SOLVEPNP_ITERATIVE
                )

        # conversions
        self.cvt_ = Conversions(
                self.K_, self.D_,
                self.T_c2b_,
                det=det,
                des=des
                )

        # data
        self.graph_ = VGraph(self.cvt_)
        self.map_ = VMap(self.cvt_, des)
        # exchange pointers
        self.graph_.set_map( self.map_ )
        self.map_.set_graph( self.graph_ )

        # processing handles

        self.track_ = Tracker(self.pLK_)
        self.ba_ = BASolver(
                self.cvt_.K_, self.cvt_.Ki_,
                self.cvt_.T_c2b_, self.cvt_.T_b2c_,
                self.pBA_)
        self.pnp_ = PNPSolverRANSAC(self.cvt_,
                thresh=1.0, prob=0.999)

    def run_PNP(self, pt3, pt2, pose,
            ax=None, msg=''):
        # TODO : fill in more RANSAC Parameters from pPNP
        pnp_it, pose_pnp, _ = self.pnp_(pt3, pt2,
                guess=pose, max_it=128)
        print_ratio('pnp it', pnp_it, 128)

        if pose_pnp is not None:
            # valid pose found, proceed
            pose = M.lerp(pose, pose_pnp, 0.5)

            if ax is not None:
                V.draw_pose(ax['main'], pose, label='pnp')

        return msg, pose

    def __call__(self, img, dt, ax=None):
        msg = ''

        # estimate current pose and update state to current index
        pose_p, pose_c, sc = self.graph_.predict(dt, commit=True) # with commit:=True, current pose index will also be updated
        index = self.graph_.index # << index should NOT change after predict()
        self.graph_.set_data_from( img ) # << this must be called after predict()

        # query graph for processing data (default : neighboring frames only)
        img_p, kpt_p, des_p, pt2_p, rsp_p = self.graph_.get_data(-2) # previous
        img_c, _, _, _, _ = self.graph_.get_data(-1) # current
        # NOTE : as of right now, pt2_c, rsp_c are propagated from pt2_p and rsp_p
        # and are not the results from self.graph_.get_data()

        # track points for PnP + LMK clearing
        idx_l, pt2_l0 = self.map_.get_track()
        li1 = len(idx_l) # number of currently tracking landmarks
        pt3_l = self.map_.pos[idx_l] # use pos() with caution; performs computation rather than lookup
        pt2_l1, ti = self.track_(img_p, img_c, pt2_l)

        # update landmark tracking information
        o_nmsk = np.ones(li1, dtype=np.bool)
        o_nmsk[ti] = False
        self.map_.pt2[idx_l[ti]] = pt2_l1
        self.map_.untrack(idx_l[o_nmsk])

        # refine kf pose with PnP
        msg, pose_c = self.run_PNP(pt3_l, pt2_l1,
                pose_c, ax=ax, msg=msg)







