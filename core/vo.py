from utils.conversions import Conversions
from core.track import Tracker
from core.match import Matcher
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
        self.match_ = Matcher(des)
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

    def track(self, pose,
            img0, img1,
            pt0, des0):
        """
        Note:
            Implements persistent visual tracking with landmarks.
            4 outcomes from track:
                [new, recovery, (suppress), old]
        Returns:
            (ptn0, ptn1), (pto0, pto1, o_iol), (ptr0, ptr1, o_irl)
        """
        q_idx = self.map_.query( pose ) # < note : high tolerance
        trk = self.map_['trk']
        desl = self.map_['des'][q_idx]
        i_trk = np.where(trk)[0]

        i_m_s0, i_m_sl = self.match_(des0, desl, **Matcher.PRESET_SOFT)
        i_m_h0, i_m_hl = self.match_(des0, desl, **Matcher.PRESET_HARD)

        # new : "fail" soft match
        # propagates pt0 from current frame
        ptn0 = pt0[M.invert_index(i_m_s0, len(des0))]
        ptn1, m_n = self.track_(img0, img1, ptn0)
        ptn0 = ptn0[m_n]
        ptn1 = ptn1[m_n]
        # TODO : potentially apply more filters here

        # old : tracked
        # propagates pt0 from track
        pto0 = ptl0[trk]
        pto1, m_o = self.track_(img0, img1, pto0)
        pto0 = pto0[m_o]
        pto1 = pto1[m_o]

        o_iol = np.where( trk & m_o ) [0]

        # recovery : "pass" hard match + untracked
        # propagates pt0 from current frame
        # i_r = [i for (i, ir) in zip(i_m_h0, i_m_hl) if not trk[q_idx][ir] ]
        msk_r = ~trk[q_idx][i_m_hl]
        i_r = i_m_h0[msk_r]
        ptr0 = pt0[i_r]
        ptr1, m_r = self.track_(img0, img1, ptr0)
        ptr0 = ptr0[m_r]
        ptr1 = ptr1[m_r]

        o_irl = np.setdiff1d(q_idx[i_m_hl], i_trk)[m_r]

        # suppress : pass soft match + tracked
        # (no_op)

        # update tracking flags
        self.map_['trk'] &= m_o
        self.map_['trk'][o_irl] = True

        # update tracking points
        self.map_['kpt'][o_iol] = pto1
        self.map_['kpt'][o_irl] = ptr1

        # update observations graph - maybe.
        self.graph_.add_obs( o_iol, pto1 ) # old
        self.graph_.add_obs( o_irl, ptr0, self.graph_.index-1) # recovery-prv
        self.graph_.add_obs( o_irl, ptr1 ) # recovery-cur

        return (ptn0, ptn1), (pto0, pto1, o_iol), (ptr0, ptr1, o_irl)

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
       
        # full tracking
        res = self.track(pose1, img0, img1, pt20, des0)
        (ptn0, ptn1), (pto0, pto1, o_iol), (ptr0, ptr1, o_irl) = res

        # refine KF pose with PnP
        msg, pose1 = self.run_PNP(
                self.map_['pos'][o_iol], pto1,
                pose1, ax, msg=msg)

        # compute refined pose information from full point correspondences
        pta0 = np.concatenate([ptn0, pto0, ptr0], axis=0)
        pta1 = np.concatenate([ptn1, pto1, ptr1], axis=0)
        i_n = np.s_[:len(ptn0)]
        i_o = np.s_[len(ptn0):-len(ptr0)]
        i_r = np.s_[-len(ptr0):]

        pose1 = self.f2f_.pose(
                pta0, pta1,
                pose0, pose1)


        # additional refinements from older frames with greater displacement
        # TODO : currently disabled while waiting for architectural decisions.
        """
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
        """

        # finalize pose
        pose1 = self.graph_.update(1, [pose1])

        # finalize point cloud
        Tb0 = M.p3_T(pose0) # b0 -> o
        Tb1 = M.p3_T(pose1)
        Tb1i = M.Ti(Tb1) # o -> b1
        Tbb = np.dot(Tb1i, Tb0)
        Tcc = cvt.Tb2Tc(Tbb)
        KP1 = cvt.K_.dot(np.eye(3,4))
        KP0 = cvt.K_.dot(Tcc[:3])
        pt31 = W.triangulate_points(
                KP1, KP0,
                pta1, pta0)

        # update map
        self.map_.append(index, pt31[i_n], pose1,
                pta0[i_n], des0[i_n], col[i_n])
        pt3 = self.cvt_.cam_to_map(pt31, pose1)
        dpt = self.cvt_.map_to_cam(pt3, self.map_['src'][o_iol])[..., 2]

        self.map_.update(o_iol,
                dpt_new=dpt[i_o],
                var_new='auto',
                hard=False
                )
        self.map_.update(o_irl,
                dpt_new=dpt[i_r]
                var_new='auto',
                hard=False
                )


