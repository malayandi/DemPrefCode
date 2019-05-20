import theano as th
import theano.tensor as tt
import numpy as np
import scipy.optimize

def extract(var):
    return th.function([], var, mode=th.compile.Mode(linker='py'))()

def shape(var):
    return extract(var.shape)

def vector(n, name=None):
    return th.shared(np.zeros(n), name=name)

def matrix(n, m, name=None):
    return th.shared(np.zeros((n, m)), name=name)

def tensor(n, m, k, name=None):
    return th.shared((np.zeros((n,m,k))), name=name)

def grad(f, x, constants=[]):
    ret = th.gradient.grad(f, x, consider_constant=constants, disconnected_inputs='warn')
    if isinstance(ret, list):
        ret = tt.concatenate(ret)
    return ret

def jacobian(f, x, constants=[]):
    sz = shape(f)
    return tt.stacklists([grad(f[i], x) for i in range(sz)])
    ret = th.gradient.jacobian(f, x, consider_constant=constants)
    if isinstance(ret, list):
        ret = tt.concatenate(ret, axis=1)
    return ret

def hessian(f, x, constants=[]):
    return jacobian(grad(f, x, constants=constants), x, constants=constants)

class Maximizer(object):
    # f is the function to optimize
    # vs is the vector to optimize over
    def __init__(self, f, vs, g={}, pre=None):
        self.pre = pre
        self.f = f
        self.vs = vs
        self.sz = [shape(v)[0] for v in self.vs]
        for i in range(1,len(self.sz)):
            self.sz[i] += self.sz[i-1]
        self.sz = [(0 if i==0 else self.sz[i-1], self.sz[i]) for i in range(len(self.sz))]
        if isinstance(g, dict):
            self.df = tt.concatenate([g[v] if v in g else grad(f, v) for v in self.vs])
        else:
            self.df = g
        self.new_vs = [tt.vector() for v in self.vs]
        self.func = th.function(self.new_vs, [-self.f, -self.df], givens=dict(zip(self.vs, self.new_vs)))
        def f_and_df(x0):
            if self.pre:
                for v, (a, b) in zip(self.vs, self.sz):
                    v.set_value(x0[a:b])
                self.pre()
            return self.func(*[x0[a:b] for a, b in self.sz])
        self.f_and_df = f_and_df
    def argmax(self, vals={}, bounds={}):
        if not isinstance(bounds, dict):
            bounds = {v: bounds for v in self.vs}
        B = []
        for v, (a, b) in zip(self.vs, self.sz):
            if v in bounds:
                B += bounds[v]
            else:
                B += [(None, None)]*(b-a)
        x0 = np.hstack([np.asarray(vals[v]) if v in vals else v.get_value() for v in self.vs])
        # print("BOUNDS", B)
        opt = scipy.optimize.fmin_l_bfgs_b(self.f_and_df, x0=x0, bounds=B)[0]
        return {v: opt[a:b] for v, (a, b) in zip(self.vs, self.sz)}
    def maximize(self, *args, **vargs):
        result = self.argmax(*args, **vargs)
        for v, res in result.items():
            v.set_value(res)
