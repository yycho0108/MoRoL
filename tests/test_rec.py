import numpy as np
import cv2
from utils import vmath as M
from utils.data import generate_data
from utils import cv_wrap as W
from viz.log import print_Rt
import viz.draw as V

from tf import transformations as tx
from scipy.spatial import Delaunay

from matplotlib import pyplot as plt

def region_index(e, w, h):
    x, y = e
    xi = (0 if (x < 0) else (2 if (x > w) else 1))
    yi = (0 if (y < 0) else (2 if (y > h) else 1))
    return (3 * yi) + xi

class EpiRect(object):
    def __init__(self, params = {}):
        self.P_ = params
        self.pEM_ = dict(
                )

        self.rmap_ = {
                0 : (1,3),
                1 : (1,0),
                2 : (2,0),
                3 : (0,3),
                4 : (-1,-1), # 'invalid'
                5 : (2,1),
                6 : (0,2),
                7 : (3,2),
                8 : (3,1)
                }

    @staticmethod
    def rh2xy(r, h, p0):
        x = p0[...,0] + r * np.cos(h)
        y = p0[...,1] + r * np.sin(h)
        return x, y

    @staticmethod
    def merge_range(ra, rb):
        da =( ra[1] - ra[0] )
        db =( rb[1] - rb[0] )
        if np.sign(da) != np.sign(db):
            return merge_range(ra, [rb[0], rb[1]])

    @staticmethod
    def resolve_hlim(
            # indices
            i0a, i1a,
            i0b, i1b,
            # data
            ha, hb,
            box
            ):

        # add a->b
        leq = F.dot(M.to_h(box[i0a]))
        hb2 = np.arctan2(leq[1], leq[0])
        leq = F.dot(M.to_h(box[i1a]))
        hb3 = np.arctan2(leq[1], leq[0])

        lim_b0 = [ha[i0b], hb[i1b]]
        lim_b1 = [hb2, hb3]

        # add b->a
        leq = F.T.dot(M.to_h(box[i0b]))
        ha2 = np.arctan2(leq[1], leq[0])
        leq = F.T.dot(M.to_h(box[i1b]))
        ha3 = np.arctan2(leq[1], leq[0])

        lim_a0 = [ha[i0a], hb[i1a]]
        lim_a1 = [ha2, ha3]

        return h0a, h1a, h0b, h1b

    def region(
            ea, h0a, h1a,
            eb, h0b, h1b,
            h0o, dro,
            mnra, mnrb,
            mxra, mxrb,
            w, h
            ):

        avra = mnra + 0.5 * dro
        avrb = mnrb + 0.5 * dro
        dha  = (h1a-h0a)
        dhb  = (h1b-h0b)
        
        dho_ = [avra*dha, avrb*dhb]
        sel  = np.argmax(dho_)
        dho  = dho_[sel]

        # inverse scale : dho --> dha,dhb
        siha = dha / dho
        sihb = dhb / dho

        rho, rro = np.mgrid[h0o:h0o+dho, :dro]
        rxa, rya = self.rh2xy(rro, h0a + (rho-h0o)*siha, ea)
        rxb, ryb = self.rh2xy(rro, h0b + (rho-h0o)*sihb, eb)

        return region_a, region_b, region_o

    def __call__(self,
            img1, img2,
            p1, p2,
            K, E=None, F=None):
        K = self.unroll()
        if E is None:
            if F is None:
                E, _ = W.E(p1, p2, cameraMatrix=K, **self.pEM_)
                _, R, t, _ = cv2.recoverPose(E, p1, p2,
                        cameraMatrix=K, 
                        )
                print('computed pose')
                print_Rt(R, t)
                F = M.E2F(E, K=K)
            else:
                E = M.F2E(E, K=K)

        # img stats
        h,w = img1.shape[:2]
        img_box = np.float32( [[0,0],[w,0],[w,h],[0,h]] )

        # compute epipoles as null-space
        e2, s, e1T = np.linalg.svd(F)
        i = np.argmin(s)
        print 'e1', M.from_h(e1T[i,:])
        print 'e2', M.from_h(e2[:,i])

        # compute epipoles
        e1 = M.from_h(np.linalg.multi_dot([K,-R.T,t.ravel()]))
        e2 = M.from_h(np.linalg.multi_dot([K,t.ravel()]))
        print 'e1', e1
        print 'e2', e2
        
        #leq = F.dot( M.to_h(p1[0])) # epiline 1->2
        #print M.uvec(leq[:2])
        #print 'leq-1', M.uvec( leq[:2] )
        #print 'leq-guess', np.arctan2(leq[1], leq[0])
        #tmp = p2[0]-e2
        #print 'gt', np.arctan2(tmp[...,1], tmp[...,0])

        # extreme displacements
        dc1 = img_box - e1[None,:]
        dc2 = img_box - e2[None,:]
        h1 = np.arctan2(dc1[:,1], dc1[:,0]) # NOTE: +cw
        h2 = np.arctan2(dc2[:,1], dc2[:,0])

        r1 = M.norm(dc1)
        r2 = M.norm(dc2)
        mxr1, mxr2 = r1.max(), r2.max()
        mnr1 = np.clip(-cv2.pointPolygonTest(img_box, tuple(e1), measureDist=True), 0, None)
        mnr2 = np.clip(-cv2.pointPolygonTest(img_box, tuple(e2), measureDist=True), 0, None)

        dr = np.maximum(mxr1-mnr1, mxr2-mnr2 )

        r1 = region_index(e1, w, h)
        r2 = region_index(e2, w, h)

        i_hb1, i_he1 = self.rmap_[r1]
        i_hb2, i_he2 = self.rmap_[r2]

        #print 'cv rec', M.uvec(cv2.computeCorrespondEpilines(
        #        np.float32( [p1[0]] ),
        #        1, F).ravel()[:2])

        out1, out2 = (mnr1 > 0), (mnr2 > 0)

        if out1 and out2:
            # both outside image
            print('both out')
            hb1, he1 = h1[i_hb1], h1[i_he1]
            if he1 < hb1: he1 += 2*np.pi

            leq = F.dot( M.to_h( img_box[i_hb1])) # epiline 1->2
            #hb2 = np.arctan2(leq[1], leq[0])
            #he2 = hb2 + h2[i_hb2] - h2[i_hb1]
            hb2, he2 = h2[i_hb2], h2[i_he2]# + (h2[i_hb2] - h2[i_hb1])
            if he2 < hb2: he2 += 2*np.pi
            # TODO : requires angle offset syncing or not??
        elif out1:
            # only ep1 outside image
            print('e1 out')
            hb1, he1 = h1[i_hb1], h1[i_he1]
            if he1 < hb1: he1 += 2*np.pi
            #hb2 = h2[i_hb1]
            leq1 = F.dot( M.to_h( img_box[i_hb1])) # epiline 1->2
            hb2 = np.arctan2(leq1[1], leq1[0]) # sync start
            leq2 = F.dot( M.to_h( img_box[i_hb2])) # epiline 1->2
            he2 = np.arctan2(leq2[1], leq2[0])# * (h2[i_he2] - h2[i_hb2]) # sync end
        elif out2:
            # only ep2 outside image
            print('e2 out')
            hb2, he2 = h2[i_hb2], h2[i_he2]
            hb1 = h1[i_hb2]
            he1 = h1[i_he2]
            #he1 = hb1 + 2 * np.pi
        else:
            print('both in')
            # both inside image
            # choose any corresponding epilines
            # + wrap 360'
            hb1 = h1[0]
            he1 = h1[0] + 2 * np.pi
            hb2 = h2[0]
            he2 = h2[0] + 2 * np.pi

        # TODO : may need to deal with flips at some point
        flip = ( np.abs( M.anorm(hb1 - hb2) ) > (np.pi/2) )
        print('flip', flip)

        #if flip:
        #    print 'flipping!'
        #    hb2,he2 = -he2,-hb2

        dh1  = (he1 - hb1)
        dh2  = (he2 - hb2)
        avr1 = mnr1 + 0.5 * dr
        avr2 = mnr2 + 0.5 * dr
        dho_ = [avr1*dh1, avr2*dh2]
        sel  = np.argmax(dho_)
        dho  = dho_[sel]

        sih1 = dh1 / dho
        sih2 = dh2 / dho

        print 'im here'
        print 'dr', dr
        print 'dho', dho

        ho, ro = np.mgrid[:dho,:dr].astype(np.float32)
        print ro.shape
        x1, y1 = self.rh2xy(mnr1+ro, hb1+ho*sih1, e1)

        #if flip:
        #    print('flipping!')
        #    x2, y2 = self.rh2xy(mnr2+ro, he2-ho*sih2, e2)
        #else:
        x2, y2 = self.rh2xy(mnr2+ro, hb2+ho*sih2, e2)

        img1_r = cv2.remap(img1, x1, y1, cv2.INTER_LINEAR)
        img2_r = cv2.remap(img2, x2, y2, cv2.INTER_LINEAR)

        ## WARP PREVIEW ##
        print x1.shape, x1.size
        idx = np.random.choice(x1.size, size=1024)
        plt.plot(x1.ravel()[idx], y1.ravel()[idx], 'bx', label='img1', alpha=0.5)
        plt.plot(x2.ravel()[idx], y2.ravel()[idx], 'g+', label='img2', alpha=0.5)
        plt.plot(ro.ravel()[idx], ho.ravel()[idx], 'r.', label='warp', alpha=0.5)
        plt.plot(img_box[:,0], img_box[:,1], 'k--')

        plt.plot(e1[None,0], e1[None,1], 'co')
        plt.plot(e2[None,0], e2[None,1], 'co')

        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()

        plt.axis('equal')#, 'datalim')

        plt.legend()
        #plt.pause(0.001)
        plt.show()
        ## WARP PREVIEW END ##

        ## remap
        # img1_r = cv2.warpPolar(img1, dsize=dsize,
        #         center=tuple(e1),
        #         maxRadius = ddr,
        #         flags=cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
        #         )

        # img2_r =gg cv2.warpPolar(img2, dsize=dsize,
        #         center=tuple(e2),
        #         maxRadius = ddr,
        #         flags=cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
        #         )

        # visualize
        eimg1, eimg2 = V.draw_epilines(img1, img2, p1, p2,
                F=F,
                sample=32
                )

        cv2.line(eimg1,
                tuple(M.rint(e1)),
                tuple(M.rint(p1[0])), (255,0,0) ,3)
        cv2.line(eimg1,
                tuple(M.rint(e1)),
                tuple(M.rint(p1[1])), (0,0,255) ,3)

        cv2.line(eimg2,
                tuple(M.rint(e2)),
                tuple(M.rint(p2[0])), (255,0,0) ,3)

        cv2.line(eimg2,
                tuple(M.rint(e2)),
                tuple(M.rint(p2[1])), (0,0,255) ,3)

        #tmp = M.rint(mr * np.pi / 4)
        #return eimg1, eimg2, eimg1, eimg2
        return img1_r, img2_r, eimg1, eimg2

    def xy2rh(self, img):
        pass
    def unroll(self):
        #context.update( self.P_ )
        return self.P_['K']

def gen_obs(K, w, h, n=100):
    d = np.random.uniform(0.1, 10.0, size=n)
    pa = np.random.uniform((0,0), (w,h), size=n)
    p3 = np.linalg.inv(K)
    return (None, None, None)


def draw_structure(img_a, img_b, pa, pb):
    ta = Delaunay(pa, incremental=False)
    tb = Delaunay(pb, incremental=False)

    ma = []
    mb = []
    mska = np.zeros(img_a.shape[:2], dtype=np.uint8)
    mskb = np.zeros(img_b.shape[:2], dtype=np.uint8)

    #ta.simplices.sort(axis=1)
    #tb.simplices.sort(axis=1)
    #simplices = M.intersect2d(ta.simplices, tb.simplices)
    simplices=ta.simplices
    cols = np.random.randint(0,255,size=(len(simplices), 3))

    cv2.drawContours(img_a, M.rint(pa[simplices]), -1,
            (255,255,255), 1)

    cv2.drawContours(img_b, M.rint(pb[simplices]), -1,
            (255,255,255), 1)
def main():
    # fix random state
    seed = None
    #seed = 20402
    #seed = 40019
    #seed = 20097
    #seed = 60389
    #seed = 48139
    #seed = 12000

    # auto initialize seed if not provided
    if seed is None:
        seed = np.random.randint(0, 65536)
    print('seed was', seed)
    np.random.seed( seed ) 

    # camera intrinsics
    w,h = (640, 480)
    fx = fy = 500.0
    cx, cy = w / 2.0, h / 2.0
    K = np.float32([fx,0,cx,0,fy,cy,0,0,1]).reshape(3,3)

    # camera extrinsics
    # optical rotation matrix : +z(cam) -> +x(base)
    T_c2b = tx.euler_matrix(-np.pi/2,0,-np.pi/2)

    params = dict(
            K=K
            )
    rec = EpiRect(params)
    p3, pa, pb, pose = generate_data(
        n=64,
        max_it=128,
        K=K,
        T_c2b=T_c2b,
        s_pos=[0.1, 0.1, 1.0],
        #s_pt3=None,
        #seed=None,
        )

    Tbb = M.p3_T(pose)
    Tcc = np.linalg.multi_dot([M.Ti(T_c2b), M.Ti(Tbb), T_c2b])
    print('gt pose')
    print_Rt(Tcc[:3,:3], Tcc[:3,3:])

    # populate image from randomized point sets
    #img1 = np.tile([[[0],[255]],[[255],[0]]], (3,4,3)).astype(np.uint8)
    #img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_NEAREST)
    img1 = np.zeros(shape=(h,w,3), dtype=np.uint8)
    img2 = np.zeros(shape=(h,w,3), dtype=np.uint8)

    draw_structure(img1, img2, pa, pb)

    ##img1 = np.random.randint(0, 128, size=(6,8,3), dtype=np.uint8)
    ##img2 = np.random.randint(0, 128, size=(6,8,3), dtype=np.uint8)
    #img2 = np.tile([[[0],[255]],[[255],[0]]], (3,4,3)).astype(np.uint8)
    #img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_NEAREST)

    cols = np.random.randint(0, 255, size=(len(pa),3), dtype=np.uint8)
    for (_pa, _pb, _c) in zip(pa,pb,cols):
        cv2.circle(img1, tuple(M.rint(_pa)), 3, tuple(_c.tolist()), -1)
        cv2.circle(img2, tuple(M.rint(_pb)), 3, tuple(_c.tolist()), -1)

    imr1, imr2, ei1, ei2 = rec(img1, img2, pa, pb, K)
    cv2.imshow('imr1', imr1)
    cv2.moveWindow('imr1', 0, 0)
    cv2.imshow('imr2', imr2)
    cv2.moveWindow('imr2', w+50, 0)
    cv2.imshow('ei1', ei1)
    cv2.imshow('ei2', ei2)
    while True:
        k = cv2.waitKey(0)
        if k == 27:
            break

if __name__ == "__main__":
    main()
