import typing, sys
sys.path.insert(0,'../..')

import numpy as np
import pandas as pd
import pickle

import domain
import human
import query_generation
import sampling


DOM = domain.FetchMove(time_steps=152)

### Parameters
N_DEMOS = 5
BETA_DEMO = 0.1
N_SAMPLES_SUMM = 50000

def run() -> typing.List:
    ### Creating sampler
    sampler = sampling.Sampler(n_query=2, dim_features=DOM.feature_size, update_func="approx",
                               beta_demo=BETA_DEMO, beta_pref=10)

    ### Collecting and loading demos
    demo_names = []
    for _ in range(N_DEMOS):
        inp = input("When you are ready, input 'y', and you can start providing your demonstration.")
        if inp == "y":
            demo_names.extend(DOM.collect_dems())
    demo_path = 'dempref_demonstrations/'
    demos = [pickle.load(open(f'{demo_path}{d}', 'rb'), encoding='latin1') for d in demo_names]
    demos = [DOM.fetch_to_mujoco(x) for x in demos]
    phi_demos = [DOM.np_features(x) for x in demos]
    sampler.load_demo(np.array(phi_demos))

    ### Computing BIRL estimate of w
    samples = sampler.sample(N=N_SAMPLES_SUMM)
    birl_w = np.mean(samples, axis=0)
    birl_w = birl_w / np.linalg.norm(birl_w)
    var_w = np.var(samples, axis=0)
    print("birl_w: ", birl_w)
    print("var_w: ", var_w)

    return [birl_w]

if __name__ == "__main__":
    args = sys.argv
    user = args[1]
    name = f"results/{user}-irl"
    w_estimates = run()
    with open(name + ".pickle", 'wb') as f:
        pickle.dump(w_estimates, f)

