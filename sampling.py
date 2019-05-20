import time
from typing import List, Dict

import numpy as np
import pymc as mc
import theano as th
import theano.tensor as tt


class Sampler(object):
    def __init__(self, n_query:int, dim_features:int, update_func:str="pick_best", beta_demo:float=0.1, beta_pref:float=1.):
        """
        Initializes the sampler.

        :param n_query: Number of queries.
        :param dim_features: Dimension of feature vectors.
        :param update_func: options are "rank", "pick_best", and "approx". To use "approx", n_query must be 2. Will throw an assertion
            error otherwise.
        :param beta_demo: parameter measuring irrationality of human in providing demonstrations
        :param beta_pref: parameter measuring irrationality of human in selecting preferences
        """
        self.n_query = n_query
        self.dim_features = dim_features
        self.update_func = update_func
        self.beta_demo = beta_demo
        self.beta_pref = beta_pref

        if self.update_func=="approx":
            assert self.n_query == 2, "Cannot use approximation to update function if n_query > 2"
        elif not (self.update_func=="rank" or self.update_func=="pick_best"):
            raise Exception(update_func + " is not a valid update function.")

        # feature vectors from demonstrated trajectories
        self.phi_demos = np.zeros((1, self.dim_features))
        # a list of np.arrays containing feature difference vectors and which encode the ranking from the preference
        # queries
        self.phi_prefs = []

        self.f = None

    def load_demo(self, phi_demos:np.ndarray):
        """
        Loads the demonstrations into the Sampler.

        :param demos: a Numpy array containing feature vectors for each demonstration.
            Has dimension n_dem x self.dim_features.
        """
        self.phi_demos = phi_demos

    def load_prefs(self, phi: Dict, rank):
        """
        Loads the results of a preference query into the sampler.

        :param phi: a dictionary mapping rankings (0,...,n_query-1) to feature vectors.
        """
        result = []
        if self.update_func == "rank":
            result = [None] * len(rank)
            for i in range(len(rank)):
                result[i] = phi[rank[i]]
        elif self.update_func == "approx":
            result = phi[rank] - phi[1-rank]
        elif self.update_func == "pick_best":
            result, tmp = [phi[rank] - phi[rank]], []
            for key in sorted(phi.keys()):
                if key != rank:
                    tmp.append(phi[key] - phi[rank])
            result.extend(tmp)
        self.phi_prefs.append(np.array(result))


    def clear_pref(self):
        """
        Clears all preference information from the sampler.
        """
        self.phi_prefs = []

    def sample(self, N:int, T:int=1, burn:int=1000) -> List:
        """
        Returns N samples from the distribution defined by applying update_func on the demonstrations and preferences
        observed thus far.

        :param N: number of samples to draw.
        :param T: if greater than 1, all samples except each T^{th} sample are discarded.
        :param burn: how many samples before the chain converges; these initial samples are discarded.
        :return: list of samples drawn.
        """
        x = tt.vector()
        x.tag.test_value = np.random.uniform(-1, 1, self.dim_features)

        # define update function
        start = time.time()
        if self.update_func=="approx":
            self.f = th.function([x], tt.sum([-tt.nnet.relu(-self.beta_pref * tt.dot(self.phi_prefs[i], x)) for i in range(len(self.phi_prefs))])
                            + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)))
        elif self.update_func=="pick_best":
            self.f = th.function([x], tt.sum(
                [-tt.log(tt.sum(tt.exp(self.beta_pref * tt.dot(self.phi_prefs[i], x)))) for i in range(len(self.phi_prefs))])
                            + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)))
        elif self.update_func=="rank":
            self.f = th.function([x], tt.sum( # summing across different queries
                [tt.sum( # summing across different terms in PL-update
                    -tt.log(
                        [tt.sum( # summing down different feature-differences in a single term in PL-update
                            tt.exp(self.beta_pref * tt.dot(self.phi_prefs[i][j:, :] - self.phi_prefs[i][j], x))
                        ) for j in range(self.n_query)]
                    )
                ) for i in range(len(self.phi_prefs))])
                            + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)))
        print("Finished constructing sampling function in " + str(time.time() - start) + "seconds")

        # perform sampling
        x = mc.Uniform('x', -np.ones(self.dim_features), np.ones(self.dim_features), value=np.zeros(self.dim_features))
        def sphere(x):
            if (x**2).sum()>=1.:
                return -np.inf
            else:
                return self.f(x)
        p = mc.Potential(
            logp = sphere,
            name = 'sphere',
            parents = {'x': x},
            doc = 'Sphere potential',
            verbose = 0)
        chain = mc.MCMC([x])
        chain.use_step_method(mc.AdaptiveMetropolis, x, delay=burn, cov=np.eye(self.dim_features)/5000)
        chain.sample(N*T+burn, thin=T, burn=burn, verbose=-1)
        samples = x.trace()
        samples = np.array([x/np.linalg.norm(x) for x in samples])

        # print("Finished MCMC after drawing " + str(N*T+burn) + " samples")
        return samples


