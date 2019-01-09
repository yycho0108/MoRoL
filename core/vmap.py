import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from utils import vmath as M

# TODO : currently in the process
# of refactoring with accord to the new parametrization,
# lmk = src + d * vec.

class VMap(object):
    """
    Visual Landmarks management.

    Supported Methods:
    """
    def __init__(self, cvt, descriptor):
        # Conversions() proc handle
        self.cvt_ = cvt

        # == auto unroll descriptor parameters ==
        n_des = descriptor.descriptorSize()
        # Not 100% sure, but don't want to take risks with unsigned math
        t_des = (np.uint8 if descriptor.descriptorType() == cv2.CV_8U
                else np.float32)
        s_fac = descriptor.getScaleFactor()
        n_lvl = descriptor.getNLevels()
        # =======================================

        # define datatype
        self.lmk_t_ = np.dtype([
            # Positional data
            ('fid', np.int32),      # source frame index
            ('src', np.float32, 3), # source pose (volatile)
            ('idp', np.float32),    # inverse depth
            ('vec', np.float32, 3),    # view-vector

            # Landmark Health
            ('var', np.float32), # Depth variance
            ('cnt', np.int32),   # Number of Total observations

            # Feature Information
            ('des', np.float32, n_des),
            ('oct', np.float32),
            ('rsp', np.float32),

            # Tracking Information
            ('kpt', np.float32, 2),
            ('trk', np.bool),

            # Query Cache / Logging
            ('ang', np.float32),
            ('col', np.uint8, 3),
            ('mnd', np.float32),
            ('mxd', np.float32),

            # derived data (cache)
            ('dpt', np.float32), # depth, (1.0 / idp)
            ('pos', np.float32, 3) # position, computed from
            ])

        # container management ...
        self.capacity_ = c = 1024
        self.size_ = s = 0
        self.data_ = np.empty(c, dtype=self.lmk_t_)

        # query filtering
        # NOTE : maxd/mind scale filtering may be ORB-specific.
        self.s_fac_ = s_fac
        self.n_lvl_ = n_lvl
        self.s_pyr_ = np.power(self.s_fac_, np.arange(self.n_lvl_))

        # store index for efficient pruning
        self.pidx_ = 0

    def resize(self, c_new):
        print('-------landmarks resizing : {} -> {}'.format(self.capacity_, c_new))
        d_new = np.empty(c_new, dtype=self.lmk_t_)
        d_new[:self.size_] = self.data_[:self.size_]
        self.data_ = d_new
        self.capacity_ = c_new

    def compute_pt3(self, src,
            ray=None,
            vec=None,
            idp=None,
            dpt=None,
            ):
        """
        Compute pt3 from the following combinations:
        src + ray=(dpt=(1/idp)*vec)
        """
        if ray is None:
            if dpt is None:
                dpt = (1.0 / idp)
            ray = dpt[:,None] * vec

        pt3 = self.cvt_.cam_to_map(ray, src)
        return pt3

    def append(self,
            fid, pos, src,
            kpt, des, col):
        """
        Append with minimal information.
        All derived quantities will be computed from the provided data.

        fid = Source frame Id (required for graph lookup)
        pos = Landmark position in `src` frame
        src = Source frame pose, p3 parametrization (x,y,h)
        kpt = cv2.KeyPoint() object for feature description/tracking
        des = Feature Descriptor
        col = Landmark Color (right now, for visualization only)
        """
        # compute derived quantities
        dpt = pos[:, 2]
        idp = (1.0 / dpt)
        vec = pos * idp[:, None]
        var = 0.1 * dpt # since variance is a weighting parameter, kept at dpt
        cnt = 1 # will be broadcasted
        oct = [e.octave for e in kpt]
        rsp = [e.response for e in kpt]
        kpt = [e.pt for e in kpt] # NOTE : shadows previous kpt
        trk = True
        ang = src[2] # will be broadcasted

        #print 'pos', pos
        #print 'ray', vec * (1.0 / idp[:,None])

        pos = self.compute_pt3(src, ray=pos) # NOTE : shadows previous pos
        lsf = np.float32([self.s_pyr_[e] for e in oct])
        mxd = dpt * lsf
        mnd = mxd / self.s_pyr_[-1]

        # _append should not be called outside of this class
        # since the argument order is required to be very specific.
        # HACK : pass vars() to avoid repeating names
        self._append(vars())

    def _append(self, v):
        n = len(v['pos'])
        if self.size_ + n > self.capacity_:
            self.resize(self.capacity_ * 2)
            # retry append after resize
            self._append(v)
        else:
            # assign
            i = np.s_[self.size_:self.size_+n]
            for k in self.data_.dtype.names:
                self.data_[k][i] = v[k]
            # update size
            self.size_ += n

    def query(self, src,
            atol = np.deg2rad(30.0),
            dtol = 1.2,
            trk=False
            ):
        cvt = self.cvt_

        # unroll map query source (base frame)
        src_x, src_y, src_h = np.ravel(src)

        # filter : by view angle
        a_dif = np.abs((self['ang'] - src_h + np.pi)
                % (2*np.pi) - np.pi)
        a_msk = np.less(a_dif, atol) # TODO : kind-of magic number

        # filter : by min/max ORB distance
        pt3_s = cvt.map_to_cam(self['pos'], src)
        d_msk = np.logical_and.reduce([
            self['mnd'] / dtol <= pt3_s[:, 2],
            pt3_s[:, 2] <= self['mxd'] * dtol
            ])
        
        # filter : by visibility
        if self.size <= 0:
            # no data : soft fail
            pt2   = np.empty((0,2), dtype=np.float32)
            v_msk = np.empty((0,),  dtype=np.bool)
        else:
            pt2, v_msk = cvt.pt3_pose_to_pt2_msk(
                    self['pos'], src)

        # merge filters
        msk = np.logical_and.reduce([
            v_msk,
            a_msk,
            d_msk
            ])

        if trk:
            msk &= self.trk[:,0]
        idx = np.where(msk)[0]

        # collect results + return
        pt2 = pt2[idx]
        pos = self['pos'][idx]
        des = self['des'][idx]
        var = self['var'][idx]
        cnt = self['cnt'][idx]
        return (pt2, pos, des, var, cnt, idx)

    def update(self, idx,
            dpt_new, var_new=None,
            hard=False):
        if hard:
            # total overwrite
            self['idp'][idx] = (1.0 / dpt)
        else:
            # incorporate previous information
            idp_old = self['idp'][idx]
            var_old = self['var'][idx]
            idp_new = (1.0 / dpt_new) # << TODO : eps needed?
            print 'idp: old-new', idp_old[0], idp_new[0]
            print 'var: old-new', var_old[0], var_new[0]
            vsum = var_old + var_new

            # apply standard gaussian product
            idp = (idp_old * var_new + idp_new * var_old) / (vsum)
            print 'idp: res', idp[0]
            var = (var_old * var_new) / (vsum)

            self['idp'][idx] = idp
            self['var'][idx] = var
            self['cnt'][idx] += 1

            # update derived parameters cache
            print 'pre?', self['pos'][idx[0]]
            pos_new = self.compute_pt3(
                    src=self['src'][idx],
                    vec=self['vec'][idx],
                    idp=self['idp'][idx])

            self['pos'][idx] = pos_new
            print 'post?', pos_new[0]
            print 'post?', self['pos'][idx[0]]


    def prune(self, k=8, radius=0.05, keep_last=512):
        """
        Non-max suppression based pruning.
        set k=1 to disable  nmx. --> TODO: verify this
        """
        # TODO : Tune keep_last parameter
        # TODO : sometimes pruning can be too aggressive
        # and get rid of desirable landmarks.
        # TODO : if(num_added_landmarks_since_last > x) == lots of new info
        #             search_and_add_keyframe()

        # NOTE: choose value to suppress with
        v = self['rsp']
        # v = 1.0 / self['var']

        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(self['pos'])
        d, i = neigh.kneighbors(return_distance=True)
        # filter results by non-max suppression radius
        msk_d = np.min(d, axis=1) < radius

        # neighborhood count TODO : or np.sum() < 2?
        msk_v = np.all(v[i] <= v[:,None], axis=1)

        # protect recent observations.
        # keep latest N landmarks
        msk_t = np.arange(self.size_) > (self.size_ - keep_last)
        # and keep all landmarks added after last prune
        msk_t |= np.arange(self.size_) >= self.pidx_
        # also keep all currently tracked landmarks
        msk_t |= self['trk']

        # strong responses are preferrable and will be kept
        msk_r = np.greater(self['rsp'], 48) # TODO : somewhat magical

        # non-max results + response filter
        msk_n = (msk_d & msk_v) | (~msk_d & msk_r)

        # below expression describes the following heuristic:
        # if (new_landmark) keep;
        # else if (passed_non_max) keep;
        # else if (strong_descriptor) keep;
        #msk = msk_t | (msk_d & msk_v | ~msk_d) | (msk_r & ~msk_d)
        #msk = msk_t | ~msk_d | msk_v
        # msk = msk_t | (msk_n & (np.linalg.norm(self.pos) < 30.0)) 

        msk = msk_t | (msk_n & ( (self['dpt'] < 30.0) ) ) # +enforce landmark bounds

        sz = msk.sum() # new size
        print('Landmarks Pruning : {}->{}'.format(msk.size, sz))

        for k in self.data_.dtype.names:
            self.data_[k][:sz] = self.data_[k][:self.size_][msk]

        self.size_ = sz
        self.pidx_ = self.size_

        # return pruned indices
        return np.where(msk)[0]

    def get_track(self):
        t_idx = np.where(self['trk'])[0]
        return t_idx, self['kpt'][t_idx]

    def untrack(self, idx):
        self['trk'][idx] = False

    def __getitem__(self, k):
        return self.data_[k][:self.size_]

    def __setitem__(self, k, v):
        self.data_[k][:self.size_] = v

    @property
    def size(self):
        return self.size_

    def draw(self, ax, *args, **kwargs):
        pt3_m = M.tx3(self.cvt_.T_c2b_, self['pos'])
        ax.plot(pt3_m[:,0], pt3_m[:,1], *args, **kwargs)

def main():
    """
    VMap() Unit Test.

    Tests the following methods:
        .append()
        .update()
        .query()
        .prune()
        .get_track()
        .untrack()
        .draw()
    """

    from utils.conversions import Conversions
    from utils import defaults as D
    from matplotlib import pyplot as plt

    fig = plt.figure()
    ax  = fig.gca()

    orb = cv2.ORB_create(
        nfeatures=1024,
        scaleFactor=1.2,
        nlevels=8,
        # NOTE : scoretype here influences response-based filters.
        scoreType=cv2.ORB_FAST_SCORE,
        #scoreType=cv2.ORB_HARRIS_SCORE,
        )
    cvt = Conversions(
        D.K,
        D.D,
        D.T_c2b,
        det=orb,
        des=orb
        )
    cvt.initialize( (480, 640) )

    vmap = VMap(
            cvt=cvt,
            descriptor=orb
            )
    # test append
    src = np.random.uniform(size=3)
    #dst = np.random.uniform(size=3)
    dst = src


    # generate random image + forge data
    img = np.random.uniform(low=0, high=255,
            size=(480,640,3)).astype(np.uint8)
    kpt = orb.detect(img)
    kpt, des = orb.compute(img, kpt)
    kpt = np.asarray(kpt, dtype=cv2.KeyPoint)

    ki = np.random.randint(0, len(des), size=32)
    kpt = kpt[ki]
    des = des[ki]

    dpt = np.random.uniform(0, 10, size=len(kpt))
    vec = np.einsum('ab,...b->...a',
            D.Ki, M.to_h([k.pt for k in kpt]))

    # pt3 is in src coord
    pt3 = dpt[:,None] * vec
    pt3_g = np.random.normal(dpt[:,None], scale=1.0) * vec
    # pt3_c0 is in camera odom coord
    pt3_c0 = cvt.cam_to_map(pt3, src)
    # pt3_m is in base_link map coord
    pt3_m = M.tx3(cvt.T_c2b_, pt3_c0)

    print('pt3-gt', pt3_c0[0])
    ax.plot(pt3_m[:,0], pt3_m[:,1], 'go', label='true')
    col = [img[int(k.pt[1]), int(k.pt[0])] for k in kpt]

    vmap.append(0, pt3_g, src,
            kpt, des, col)
    vmap.draw(ax, 'r^', label='pre')

    q = vmap.query(src=dst)
    #(pt2, pt3, des, var, cnt, idx) = q
    idx = q[-1]
    dpt = cvt.map_to_cam(pt3_c0[idx], dst)[:, 2]
    var = 0.01 * dpt
    vmap.update(idx, np.random.normal(dpt, scale=0.0), var)
    vmap.draw(ax, 'bx', label='post')

    vmap.prune(keep_last=0)
    vmap.untrack( np.random.randint(0, 32, 16) )
    print vmap.get_track()

    vmap.prune(keep_last=0)
    vmap.draw(ax, 'c.', label='post-prune')

    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
