import numpy as np
from sklearn.neighbors import NearestNeighbors

from utils import vmath as M
from utils import cv_wrap as W
import cv2
from viz.log import print_Rt

def resolve_Rt(R0, R1, t0, t1, alpha=0.5, guess=None):
    # TODO : deal with R0/R1 disagreement?
    # usually not a big issue.
    if np.dot(t0.ravel(), t1.ravel()) < 0:
        # disagreement : choose
        if guess is not None:
            # reference
            R_ref, t_ref = guess

            # precompute norms
            d_ref = M.norm(t_ref)
            d0 = M.norm(t0)
            d1 = M.norm(t1)

            # compute alignment score
            score0 = np.dot(t_ref.ravel(), t0.ravel()) / (d_ref * d0)
            score1 = np.dot(t_ref.ravel(), t1.ravel()) / (d_ref * d1)
        else:
            # reference does not exist, choose smaller t
            score0 = np.linalg.norm(t0)
            score1 = np.linalg.norm(t1)

        idx = np.argmax([score0, score1])

        return [(R0,t0), (R1,t1)][idx]
    else:
        # agreement : interpolate
        # rotation part
        R = M.rlerp(R0, R1, alpha)
        # translation part
        t = M.lerp(t0, t1, alpha)
        return (R, t)

def recover_pose_perm(perm, K,
        pt1, pt2,
        z_min = np.finfo(np.float32).eps,
        z_max = np.inf,
        return_index=False,
        log=False,
        threshold=0.8,
        guess=None
        ):
    P1 = np.eye(3,4)
    P2 = np.eye(3,4)

    sel   = 0
    scores = [0.0 for _ in perm]
    msks = [None for _ in perm]
    pt3s = [None for _ in perm]
    ctest = -np.inf

    for i, (R, t) in enumerate(perm):
        # Compute Projection Matrix
        P2[:3,:3] = R
        P2[:3,3:] = t.reshape(3,1)
        KP1 = K.dot(P1) # NOTE : this could be unnecessary, idk.
        KP2 = K.dot(P2)

        # Triangulate Points
        pt3 = W.triangulate_points(KP1, KP2, pt1, pt2)
        pt3_a = pt3
        pt3_b = M.tx3(P2, pt3)

        # apply z-value (depth) filter
        za, zb = pt3_a[:,2], pt3_b[:,2]
        msk_i = np.logical_and.reduce([
            z_min < za,
            za < z_max,
            z_min < zb,
            zb < z_max
            ])
        c = msk_i.sum()

        # store data
        pt3s[i] = pt3_a # NOTE: a, not b
        msks[i] = msk_i
        scores[i] = ( float(msk_i.sum()) / msk_i.size)

        if log:
            print('[{}] {}/{}'.format(i, c, msk_i.size))
            print_Rt(R, t)

    # option one: compare best/next-best
    sel = np.argmax(scores)

    if guess is not None:
        # -- option 1 : multiple "good" estimates by score metric
        # here, threshold = score
        # soft_sel = np.greater(scores, threshold)
        # soft_idx = np.where(soft_sel)[0]
        # do_guess = (soft_sel.sum() >= 2)
        # -- option 1 end --

        # -- option 2 : alternative next estimate is also "good" by ratio metric
        # here, threshold = ratio
        next_idx, best_idx = np.argsort(scores)[-2:]
        soft_idx = [next_idx, best_idx]
        if scores[best_idx] >= np.finfo(np.float32).eps:
            do_guess = (scores[next_idx] / scores[best_idx]) > threshold
        else:
            # zero-division protection
            do_guess = False
        # -- option 2 end --

        soft_scores = []
        if do_guess:
            # TODO : currently, R-guess is not supported.
            R_g, t_g = guess
            t_g_u = M.uvec(t_g.ravel()) # convert guess to uvec
            
            for i in soft_idx:
                # filter by alignment with current guess-translational vector
                R_i, t_i = perm[i]
                t_i_u = M.uvec(t_i.ravel())
                score_i = t_g_u.dot(t_i_u)
                soft_scores.append(score_i)

            # finalize selection
            sel = soft_idx[ np.argmax(soft_scores) ]
            unsel = soft_idx[ np.argmin(soft_scores) ] # NOTE: log-only

            if True: # TODO : swap with if log:
                print('\t\tresolving ambiguity with guess:')
                print('\t\tselected  i={}, {}'.format(sel, perm[sel]))
                print('\t\tdiscarded i={}, {}'.format(unsel, perm[unsel]))

    R, t = perm[sel]
    msk = msks[sel]
    pt3 = pt3s[sel][msk]
    n_in = msk.sum()

    if return_index:
        return n_in, R, t, msk, pt3, sel
    else:
        return n_in, R, t, msk, pt3

def recover_pose(E, K,
        pt1, pt2,
        z_min = np.finfo(np.float32).eps,
        z_max = np.inf,
        threshold=0.8,
        guess=None,
        log=False
        ):
    R1, R2, t = cv2.decomposeEssentialMat(E)
    perm = [
            (R1, t),
            (R2, t),
            (R1, -t),
            (R2, -t)]
    return recover_pose_perm(perm, K,
            pt1, pt2,
            z_min, z_max,
            threshold=threshold,
            guess=guess,
            log=log
            )

def pts_nmx(
        pt, pt_ref,
        val, val_ref,
        k=16,
        radius=1.0, # << NOTE : supply valid radius here when dealing with 2D Data
        thresh=1.0
        ):
    # NOTE : somewhat confusing;
    # here suffix c=camera, l=landmark.
    # TODO : is it necessary / proper to take octaves into account?
    if len(pt_ref) < k:
        # Not enough references to apply non-max with.
        return np.arange(len(pt))

    # compute nearest neighbors
    neigh = NearestNeighbors(n_neighbors=k, radius=radius)
    neigh.fit(pt_ref)

    # NOTE : 
    # radius_neighbors would be nice, but indexing is difficult to use
    # res = neigh.radius_neighbors(pt_new, return_distance=False)
    d, i = neigh.kneighbors(pt, return_distance=True)

    # too far from other landmarks to apply non-max
    msk_d = (d.min(axis=1) >= radius)
    # passed non-max
    msk_v = np.all(val_ref[i] < thresh*val[:,None], axis=1) # 

    # format + return results
    msk = (msk_d | msk_v)
    idx = np.where(msk)[0]
    print_ratio('non-max', len(idx), msk.size)
    return idx
