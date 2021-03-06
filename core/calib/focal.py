import numpy as np

def solve_focal(s1, s2, u1, u2, u3, v1, v2, v3):
    """ strum's focal calibration """
    # TODO : maybe make it more efficient
    u11,u12,u13 = u1[...,0], u1[...,1], u1[...,2]
    u21,u22,u23 = u2[...,0], u2[...,1], u2[...,2]
    u31,u32,u33 = u3[...,0], u3[...,1], u3[...,2]

    v11,v12,v13 = v1[...,0], v1[...,1], v1[...,2]
    v21,v22,v23 = v2[...,0], v2[...,1], v2[...,2]
    v31,v32,v33 = v3[...,0], v3[...,1], v3[...,2]

    a = s1
    b = s2

    fsq1 = - u23*v13*(a*u13*v13+b*u23*v23) / (a*u13*u23*(1-v13*v13)+b*v13*v23*(1-u23*u23))
    fsq2 = - u13*v23*(a*u13*v13+b*u23*v23) / (a*v13*v23*(1-u13*u13)+b*u13*u23*(1-v23*v23))

    alpha = a*a*(1-u13*u13)*(1-v13*v13)-b*b*(1-u23*u23)*(1-v23*v23)
    beta  = a*a*(u13*u13+v13*v13-2*u13*u13*v13*v13)-b*b*(u23*u23+v23*v23-2*u23*u23*v23*v23)
    gamma = a*a*u13*u13*v13*v13-b*b*u23*u23*v23*v23

    fsq3 = (-beta + np.sqrt(beta**2 - 4*alpha*gamma)) / (2*alpha)
    fsq4 = (-beta - np.sqrt(beta**2 - 4*alpha*gamma)) / (2*alpha)

    fsq = np.concatenate([fsq1.ravel(), fsq2.ravel(), fsq3.ravel(), fsq4.ravel()])

    # filter out invalid solutions
    msk = np.logical_and.reduce([
        np.isfinite(fsq),
        np.isreal(fsq),
        fsq > 0,
        ])
    foc = np.sqrt( fsq[msk] )

    np.save('/tmp/focs.npy', foc)
    #plt.hist(foc)
    #plt.show()

    return np.median( foc )

class FocalSolverStrum(object):
    def __init__(self, w, h):
        self.w_ = w
        self.h_ = h

    def __call__(self, Fs):
        w, h = self.w_, self.h_
        gl, gr = np.eye(3), np.eye(3)
        gl[2,0] = w / 2.0
        gl[2,1] = h / 2.0
        gr[0,2] = w / 2.0
        gr[1,2] = h / 2.0

        Gs = np.einsum('ab,...bc,cd',gl,Fs,gr) # 'semi-calibrated' space
        U, S, Vt = np.linalg.svd(Gs)
        u1, u2, u3 = U[...,:,0], U[...,:,1], U[...,:,2]
        v1, v2, v3 = Vt[...,0,:], Vt[...,1,:], Vt[...,2,:]
        s1, s2  = S[...,0], S[...,1]
        f = solve_focal(s1, s2, u1, u2, u3, v1, v2, v3)
        return f
