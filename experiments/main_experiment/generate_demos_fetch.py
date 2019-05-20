import sys, typing
sys.path.insert(0, '../..')

import pickle

import numpy as np

import domain
import sampling

QUERY_LENGTH = 5
UPDATE_FUNC = "approx"
BETA_DEMO = 0.1
BETA_PREF = 5

N_QUERY = 2

def generate_demos(dom: domain.Domain, weight: typing.List, name: str):
    t = dom.simulate(weight, query_length=QUERY_LENGTH, iter_count=10)

    sampler = sampling.Sampler(n_query=N_QUERY, dim_features=dom.feature_size, update_func=UPDATE_FUNC, beta_demo=BETA_DEMO, beta_pref=BETA_PREF)
    sampler.load_demo(dom.np_features(t))
    samples = sampler.sample(50000)
    mean_w = np.mean(samples, axis=0)
    mean_w = mean_w / np.linalg.norm(mean_w)
    print("w = " + str(mean_w))
    m = np.mean([np.dot(w, weight) / np.linalg.norm(w) / np.linalg.norm(weight) for w in samples])
    print("m = " + str(m))

    with open(f"fetch_demos/{name}.pickle", 'wb') as f:
        pickle.dump(t, f)

if __name__ == "__main__":
    args = sys.argv
    D = args[1]

    if D == "fetch_move":
        DOM = domain.FetchMove(time_steps=150)
        TRUE_WEIGHT = [-0.3, -0.3, 0.9]
    else:
        print(f"No domain named {D}.")

    generate_demos(DOM, TRUE_WEIGHT, D)
