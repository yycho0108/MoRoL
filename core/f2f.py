from utils import vmath as M
from utils import cv_wrap as W

class F2F(object):
    def __init__(self, track):
        self.track = track

    def run_GP(self):
        pass

    def run_EM(self):
        null_result = (
                np.empty((0,), dtype=np.int32), # index
                np.empty((0,3), dtype=np.float32), # points
                guess # transformation
                )

        if no_gp:
            # pre-filter by ymin
            y_gp = self.y_GP
            # NOTE: is it necessary to also check pt2_u_p?
            # probably gives similar results; skip.
            ngp_msk = (pt2_u_c[:,1] <= y_gp)
            ngp_idx = np.where(ngp_msk)[0]
            pt2_u_c = pt2_u_c[ngp_idx]
            pt2_u_p = pt2_u_p[ngp_idx]

        if len(pt2_u_c) <= 5:
            return null_result

        # EXPERIMENTAL : least-squares
        # (R, t), pt3 = solve_TRI_fast(pt2_u_p, pt2_u_c,
        #         self.cvt_.K_, self.cvt_.Ki_,
        #         self.cvt_.T_b2c_, self.cvt_.T_c2b_,
        #         guess)
        # return np.arange(len(pt2_u_p)), pt3, (R,t)

        # == opt 1 : essential ==
        # NOTE ::: findEssentialMat() is run on ngp_idx (Not tracking Ground Plane)
        # Because the texture in the test cases were repeatd,
        # and was prone to mis-identification of transforms.
        E, msk_e = cv2.findEssentialMat(pt2_u_c, pt2_u_p, self.K_,
                **self.pEM_)
        msk_e = msk_e[:,0].astype(np.bool)
        idx_e = np.where(msk_e)[0]
        print_ratio('e_in', len(idx_e), msk_e.size)
        F = self.cvt_.E_to_F(E)
        # == essential over ==

        if len(idx_e) < 16: #TODO : magic number
            # insufficient # of points -- abort
            return null_result

        if self.flag_ & ClassicalVO.VO_USE_HOMO:
            # opt 2 : homography
            H, msk_h = cv2.findHomography(pt2_u_c, pt2_u_p,
                    method=self.pEM_['method'],
                    ransacReprojThreshold=self.pEM_['threshold']
                    )
            msk_h = msk_h[:,0].astype(np.bool)
            idx_h = np.where(msk_h)[0]
            print_ratio('h_in', len(idx_h), msk_h.size)

            # compare errors
            sH, msk_sh = score_H(pt2_u_c, pt2_u_p, H, self.cvt_)
            sF, msk_sf = score_F(pt2_u_c, pt2_u_p, F, self.cvt_)

            r_H = (sH / (sH + sF))
            print_ratio('RH', sH, sH+sF)

        use_h = False
        if self.flag_ & ClassicalVO.VO_USE_HOMO:
            # TODO : "magic" determinant number based on ORB-SLAM Paper
            use_h = (r_H > 0.45)

        # TODO : option : filter based on input guess R,t
        # maybe a good idea.

        idx_p = None
        if use_h:
            # use Homography Matrix for pose
            #idx_h = idx_h[msk_sh]
            res_h, Hr, Ht, Hn = cv2.decomposeHomographyMat(H, self.K_)
            print Hr[0], Ht[0], np.linalg.norm(Hr[0]), np.linalg.norm(Ht[0])
            Ht = np.float32(Ht)

            # NOTE: still don't know why Ht is not normalized.
            Ht /= np.linalg.norm(Ht, axis=(1,2), keepdims=True)

            perm = zip(Hr,Ht)
            n_in, R, t, msk_r, pt3 = recover_pose_from_RT(perm, self.K_,
                    pt2_u_c[idx_h], pt2_u_p[idx_h], guess=guess,
                    log=False)
            print_ratio('homography', len(idx_h), msk_h.size)

            idx_p = idx_h
        else:
            # use Essential Matrix for pose
            # TODO : specify z_min/z_max?
            n_in, R, t, msk_r, pt3 = recover_pose(E, self.K_,
                    pt2_u_c[idx_e], pt2_u_p[idx_e], guess=guess,
                    log=False
                    )
            # least-squares refinement
            #(R, t), pt3 = solve_TRI(pt2_u_p[idx_e[msk_r]], pt2_u_c[idx_e[msk_r]],
            #        self.cvt_.K_, self.cvt_.Ki_,
            #        self.cvt_.T_b2c_, self.cvt_.T_c2b_,
            #        guess = (R,t) )
            #pt3 = pt3.T
            print_ratio('essentialmat', len(idx_e), msk_e.size)
            idx_p = idx_e

        # idx_r = which points were used for pose reconstruction
        pt3 = pt3.T
        idx_r = np.where(msk_r)[0]
        print_ratio('triangulation', len(idx_r), msk_r.size)

        idx_in = idx_p[idx_r] # overall, which indices were used?

        return idx_in, pt3, (R, t)


    def __call__(self,
            img0, img1,
            pose0, pose1,
            des1, pt21,
            pt3=None,
            msk3=None,
            alpha=0.5,
            ref=1,
            scale=None
            ):
        """
        High alpha = bias toward new estimates
        """
        if ref == 0:
            # (des-pt2), (pt3-msk) all refer to img0

            # NOTE : because it is flipped,
            # pt3 input must be in coord0

            # flip
            res = self.__call__(img1, img0, pose1, pose0,
                    des1, pt21, pt3, msk3, alpha, ref=1, scale=scale)

            sc, pt3, msk3, (o_pt21, o_pt20, o_idx), (o_R, o_t) = res
            # o_R/o_t is a transform from coord0 to coord1

            # pt3 is in coord. frame of 0; must convert to coord1
            pt3 = pt3.dot(o_R.T) + o_t.T

            # NOW flip o_R/o_t (NOTE : this must come after pt3 inversion)
            # TODO : replace below with more efficient computation

            o_R, o_t = M.Rti(o_R, o_t)

            return sc, pt3, msk3, (o_pt20, o_pt21, o_idx), (o_R, o_t)

        # ^^ TODO : also input pt3 cov?
        # NOTE :  pt21 **MAY** contain landmark information later.
        # I think that might be a better idea.

        if pt3 is None:
            # construct initial pt3 guess if it doesn't exist
            pt3  = np.zeros((len(pt21),3), dtype=np.float32 )
            msk3 = np.zeros((len(pt21)), dtype=np.bool)
        idx3 = np.where(msk3)[0]
        # compose initial dR&dt guess
        R_c0, t_c0 = self.pp2RcTc(pose0, pose1)

        sc0 = np.linalg.norm(t_c0)

        # track
        # NOTE: distort=false assuming images and points are all pre-undistorted
        pt20_G = pt21.copy() # expected pt2 locations @ pose0

        if False:#len(idx3) > 0:
            # fill in guesses if 3D information for pts exists
            pt20_G[idx3] = W.project_points(
                    # project w.r.t pose0.
                    # this works because T_c0 (R_c0|t_c0)
                    # represents the transform that takes everything to pose0 coordinates.
                    pt3[idx3],
                    cv2.Rodrigues(R_c0)[0], # rvec needs to be formatted as such.
                    t_c0.ravel(), # TODO : ravel needed?
                    cameraMatrix=self.K_,
                    distCoeffs=self.D_*0,
                    )

        pt20, idx_t = self.track(img1, img0, pt21, pt2=None) # NOTE: track backwards

        # TODO : FM Correction with self.run_fm_cor() ??
        F = None
        ck0 = False
        ck1 = False
        pt20_tmp = None
        pt21_tmp = None

        if self.flag_ & ClassicalVO.VO_USE_FM_COR:
            # correct Matches by RANSAC consensus
            F, msk_f = cv2.findFundamentalMat(
                    pt21[idx_t],
                    pt20[idx_t],
                    method=self.pEM_['method'],
                    param1=self.pEM_['threshold'],
                    param2=self.pEM_['prob'],
                    )
            msk_f = msk_f[:,0].astype(np.bool)
            print_ratio('FM correction', msk_f.sum(), msk_f.size)

            # retro-update corresponding indices
            # to where pt2_u_p will be, based on idx_f
            idx_t   = idx_t[msk_f]

            # NOTE : invalid to apply undistort() after correction
            # NOTE : below code will work, but validity is questionable.
            pt21_f, pt20_f = cv2.correctMatches(F,
                    pt21[None,idx_t],
                    pt20[None,idx_t])
            pt21_f = np.squeeze(pt21_f, axis=0)
            pt20_f = np.squeeze(pt20_f, axis=0)

            ## EPILINES VIZ ##
            # idx_t_v = np.random.choice(idx_t, size=32)

            # img1v = img1.copy()
            # img2v = img0.copy()
            # pt2_1v = pt21[idx_t_v]
            # pt2_2v = pt20[idx_t_v]

            # lines1 = cv2.computeCorrespondEpilines(pt2_2v.reshape(-1,1,2), 2, F)\
            #         .reshape(-1,3)
            # lines2 = cv2.computeCorrespondEpilines(pt2_1v.reshape(-1,1,2), 1, F)\
            #         .reshape(-1,3)

            # efig = self.fig_['epi']
            # eax = efig.gca()
            # 
            # cols = [tuple(np.random.randint(0,255,3).tolist()) for _ in idx_t_v]

            # drawlines(img1v, img2v, lines1, pt2_1v, pt2_2v, cols)
            # drawlines(img2v, img1v, lines2, pt2_2v, pt2_1v, cols)

            # eax.imshow( np.concatenate([img1v,img2v], axis=1))

            # plt.pause(0.001)

            ## -- will sometimes return NaN.
            ck0 = np.all(np.isfinite(pt20_f))
            ck1 = np.all(np.isfinite(pt21_f))

            if ck0 and ck1:
                pt20_tmp = pt20[idx_t]
                pt21_tmp = pt21[idx_t]
                pt20[idx_t] = pt20_f
                pt21[idx_t] = pt21_f

        # stage 1 : EM
        res = self.run_EM(pt21[idx_t], pt20[idx_t], no_gp=False, guess=(R_c0, t_c0) )
        idx_e, pt3_em_u, (R_em, t_em_u) = res # parse run_EM, no scale info
        t_em_u /= np.linalg.norm(t_em_u) # make sure uvec
        idx_e = idx_t[idx_e]
        # ^ note pt3_em_u in camera (pose1) coord

        # stage 2 : GP
        # guess based on em or c0 ?? Is it double-dipping to use R_em/t_em for GP?
        res = self.run_GP(pt21[idx_t], pt20[idx_t], sc0, guess=(R_em, t_em_u) )
        H, sc2, (R_gp, t_gp_u), (pt3_gp_u, idx_g) = res # parse run_GP
        t_gp_u /= np.linalg.norm(t_gp_u)
        if idx_g is not None:
            idx_g = idx_t[idx_g]
        # ^ note pt3_gp_u also in camera (pose1) coord

        # stage 2.5 : Restore pt20/pt21
        if self.flag_ & ClassicalVO.VO_USE_FM_COR:
            if ck0 and ck1:
                pt20[idx_t] = pt20_tmp
                pt21[idx_t] = pt21_tmp

        # stage 3 : resolve scale based on guess + GP measurement

        # interpolation factor between
        # ground-plane estimates vs. essentialmat estimates
        # high value = high GP trust
        alpha_gve = 0.5

        # resolve scale based on EM / GP results
        # NOTE : sc0 based on ekf/ukf; sc2 based on ground plane.

        sc = lerp(sc0, sc2, alpha)
        if scale is not None:
            # incorporate input scale information
            sc = lerp(scale, sc, alpha)

        # prepare observation mask
        o_msk = np.zeros((len(pt21)), dtype=np.bool)

        # 1. resolve pose observation
        o_R, o_t = resolve_Rt(R_em, R_gp, t_em_u*sc, t_gp_u*sc,
                alpha=alpha_gve,
                guess=(R_c0, t_c0))

        # 2. fill in pt3 information + mark indices
        # ( >> TODO << : incorporate confidence information )
        # NOTE: can't use msk3 for o_msk, which is aggregate info.

        # == VIZ : em/gp points comparison ==
        #_, i_e, i_g = np.intersect1d(idx_e, idx_g,
        #        return_indices=True) # idx_e[i_e] == idx_g[i_g]
        #tfig = self.fig_['pt3']
        #tax  = tfig.gca(projection='3d')
        #tax.cla()
        #tax.plot(pt3_em_u[i_e,0], pt3_em_u[i_e,1], pt3_em_u[i_e,2], 'rx', label='em')
        #tax.plot(pt3_gp_u[i_g,0], pt3_gp_u[i_g,1], pt3_gp_u[i_g,2], 'b+', label='gp')
        #tax.legend()
        #plt.pause(0.001)
        #print 'discrepancy >>', t_em_u, t_gp_u
        # == VIZ END ==

        if pt3_em_u is not None:
            pt3_em = pt3_em_u * sc
            pt3[idx_e] = np.where(
                    msk3[idx_e,None],
                    lerp(pt3[idx_e], pt3_em, alpha),
                    pt3_em)
            msk3[idx_e] = True
            o_msk[idx_e] = True

        if pt3_gp_u is not None:
            # NOTE: overwrites idx_e results with idx_g
            pt3_gp = pt3_gp_u * sc
            pt3[idx_g] = pt3_gp 
            pt3[idx_g] = np.where(
                    msk3[idx_g,None],
                    lerp(pt3[idx_g], pt3_gp, alpha),
                    pt3_gp)
            msk3[idx_g] = True
            o_msk[idx_g] = True

        # 2.  parse indices
        o_idx = np.where(o_msk)[0]
        o_pt20, o_pt21 = pt20[o_idx], pt21[o_idx]

        # NOTE : final result
        # 1. refined 3d positions + masks,
        # 2. observation of the points at the respective poses,
        # 3. observation of the relative pose from p0->p1. NOTE: specified in camera coordinates.
        return sc, pt3, msk3, (o_pt20, o_pt21, o_idx), (o_R, o_t)
