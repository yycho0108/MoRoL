from utils.conversions import Conversions
from core.track import Tracker
from core.vgraph import VGraph
from core.vmap import VMap

from utils import vmath as M
from utils import cv_wrap as W
from utils.common import pts_nmx

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
        self.pLK_ = None
        self.pPNP_ = dict(
                iterationsCount=256,
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

        # processing handles
        self.track_ = Tracker(self.pLK_)
        self.ba_ = BASolver(
                self.cvt_.K_, self.cvt_.Ki_,
                self.cvt_.T_c2b_, self.cvt_.T_b2c_,
                self.pBA_)
        self.pnp_ = PNPSolverRANSAC(self.cvt_,
                thresh=self.pPNP_['reprojectionError'],
                prob=self.pPNP_['confidence'])
        self.f2f_ = None
        self.f2m_ = None

    def initialize(self, img0, scale0,
            x0=None, P0=None):
        # 0. initialize processing handles
        self.cvt_.initialize(img0.shape)

        # 1. initialize graph + data cache
        self.graph_.initialize(x0, P0)
        self.graph_.process(img0)

        # 2. initialize scale
        self.scale0_ = scale0
        self.use_s0_ = True

    def run_PNP(self, pt3, pt2, pose,
            ax=None, msg=''):
        # TODO : fill in more RANSAC Parameters from pPNP
        pnp_it, pose_pnp, _ = self.pnp_(pt3, pt2,
                guess=pose,
                max_it=self.pPNP_['iterationsCount'])
        print_ratio('pnp it', pnp_it, self.pPNP_['iterationsCount'])

        if pose_pnp is not None:
            # valid pose found, proceed
            pose = M.lerp(pose, pose_pnp, 0.5)
            if ax is not None:
                V.draw_pose(ax['main'], pose, label='pnp')
        return msg, pose

    def __call__(self, img, dt, ax=None):
        msg = ''
        cvt = self.cvt_

        # estimate current pose and update state to current index
        pose0, pose1, sc = self.graph_.predict(dt, commit=True) # with commit:=True, current pose index will also be updated
        index = self.graph_.index # << index should NOT change after predict()
        self.graph_.process( img ) # << this must be called after predict()

        # query graph for processing data (default : neighboring frames only)
        img0, kpt0, des0, pt20, rsp0 = self.graph_.get_data(-2) # previous
        img1, _, _, _, _ = self.graph_.get_data(-1) # current
        # NOTE : kpt1,des1,pt21,rsp1 ... are propagated from previous frame,
        # And are not used until the next frame arrives.

        # track points for PnP + LMK clearing
        idx_l, pt2_l0 = self.map_.get_track()
        li1 = len(idx_l) # number of currently tracking landmarks
        pt2_l1, ti = self.track_(img0, img1, pt2_l0)

        # update landmark tracking information
        o_nmsk = np.ones(li1, dtype=np.bool)
        o_nmsk[ti] = False
        self.map_['pt2'][idx_l[ti]] = pt2_l1
        self.map_.untrack(idx_l[o_nmsk])

        # apply successful tracking indices
        pt2_l0 = pt2_l0[ti]
        pt2_l1 = pt2_l1[ti]
        idx_l = idx_l[ti]
        rsp_l  = self.map_['rsp'][idx_l]
        pt3_l = self.map_['pos'][idx_l]

        # add observation edge to graph
        self.graph_.add_obs(idx_l, pt2_l1)

        # refine kf pose with PnP
        msg, pose1 = self.run_PNP(pt3_l, pt2_l1,
                pose1, ax=ax, msg=msg)

        # track additional points
        pt21, idx_t = self.track_(img0, img1, pt20)

        # apply successful tracking indices
        kpt0 = kpt0[idx_t]
        des0 = des0[idx_t]
        rsp0 = rsp0[idx_t]
        pt20 = pt20[idx_t]
        pt21 = pt21[idx_t]

        # compute refined pose information from point correspondences
        pt20_a = np.concatenate([pt20, pt2_l0], axis=0)
        pt21_a = np.concatenate([pt21, pt2_l1], axis=0)
        pose1, idx_p = self.f2f_.pose(
                pt20_a, pt21_a,
                pose0, pose1)

        # initialize point cloud from current pose estimates
        KP1 = cvt.K_.dot(cvt.Tb2Tc(M.p3_T(pose1))[:3])
        KP0 = cvt.K_.dot(cvt.Tb2Tc(M.p3_T(pose0))[:3])
        pt3 = W.triangulate_points(
                KP1, KP0,
                pt21, pt20)

        # additional refinements from older frames with greater displacement
        # TODO : currently disabled while waiting for architectural decisions.
        for di in []:
            # acquire data from past frames
            data = self.graph_.get_data(di)
            if data is None:
                break
            imgi, kpti, desi, pt2i, rspi = data 
            posei = self.graph_.pos[di][:3]
            KPi = cvt.K_.dot(cvt.Tb2Tc(M.p3_T(posei))[:3])

            # obtain pose estimates from point tracking correspondences
            pt2i, msk_i = self.cvt_.pt3_pose_to_pt2_msk(pt3, posei)
            idx_i = np.where(msk_i)[0]
            pt2i, ti = self.track_(img1, imgi, pt21[idx_i], pt2i[idx_i])
            idx_i = idx_i[ti] # apply tracking indices

            if len(idx_i) > 16:
                pose1, idx_p = self.f2f_.pose(
                        pt2i[idx_i], pt21[idx_i],
                        posei, pose1,
                        w=0.25) # TODO : tune alpha? dynamic?
                pt3[idx_i] = M.lerp(
                        pt3[idx_i],
                        W.triangulate_points(
                            KP1, KPi,
                            pt21[idx_i], pt2i[idx_i]
                            ),
                        w=0.25) # TODO : tune alpha? dynamic?

        # finalize pose
        pose1 = self.graph_.update(1, [pose1])

        # filter new points
        idx_f = pts_nmx(pt21, pt2_l1,
                rsp0, rsp_l,
                k=4, radius=16.0, thresh=1.0
                )

        # apply filter
        pt21 = pt21[idx_f]

        # add final landmarks
        self.landmarks_.append( new_lm_args )
