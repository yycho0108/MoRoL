class ESVSolver(object):
    def __init__(self):
        pass
    def err(self, params, cfg):
        # arguments should be supplied either through unraveled params or cfg
        data = {}
        for k,s in cfg['param']:
            data[k] = param[i0:i0+np.prod(s)].reshape(s)
        for k,v in cfg['const']:
            data[k] = v
    def __call__(self):
        pass
