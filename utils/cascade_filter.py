import numpy as np

class CascadeFilter(object):
    def __init__(self, f, d={}):
        self.f_ = f
        self.d_ = d

    # some useful functions
    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def get(x, i):
        if np.isscalar(x):
            return x
        return x[i]

    def __call__(self, idx, args):
        self.d_.update( args )
        for f, ai in self.f_:
            a = [self.get(self.d_[i_a], idx) for i_a in ai]
            i2 = f(*a)
            idx = idx[i2]

        return idx

def main():
    f = CascadeFilter([
        [np.greater, ('x',0)],
        [np.less, ('x','w')],
        [np.greater, ('y',0)],
        [np.less, ('y','h')],
            ])

    pt = np.random.uniform((-500,-500), (640+500,480+500), size=(5,2))
    i0 = np.arange( len(pt) )

    i1 = f(i0, {
        'x' : pt[:, 0],
        'y' : pt[:, 1],
        0   : 0,
        'w' : 640,
        'h' : 480
        })

    m_ni1 = np.ones(len(pt), dtype=np.bool)
    m_ni1[i1] = False
    ni1 = np.where(m_ni1)[0]

    print pt[i1]
    print pt[ni1]

if __name__ == "__main__":
    main()
