import sys
sys.path.insert(0,'../..')

import pickle

import numpy as np

import domain
import sampling

dom = domain.LunarLander(time_steps=150, seed=77)
true_weight = [-0.4, 0.4, -0.2, -0.7]
QUERY_LENGTH = 10
UPDATE_FUNC = "rank"
BETA_DEMO = 0.1
BETA_PREF = 5

# dom = domain.Car(dt=0.1, time_steps=50, num_others=1)
# true_weight = [0.5, -0.2, 0.2, -0.7]
# QUERY_LENGTH = 10
# UPDATE_FUNC = "rank"
# BETA_DEMO = 0.1
# BETA_PREF = 5

N_QUERY = 2
DIM_FEATURES = dom.feature_size

ms = {}

for i in range(25):
    weight = np.random.uniform(-1, 1, 4)
    t = dom.simulate(weight, query_length=QUERY_LENGTH, iter_count=10)

    sampler = sampling.Sampler(n_query=N_QUERY, dim_features=DIM_FEATURES, update_func=UPDATE_FUNC, beta_demo=BETA_DEMO, beta_pref=BETA_PREF)
    sampler.load_demo(dom.np_features(t))
    samples = sampler.sample(50000)
    m = np.mean([np.dot(w, true_weight) / np.linalg.norm(w) / np.linalg.norm(true_weight) for w in samples])

    ms[m] = t

sorted = list(ms)
sorted.sort()

worst = sorted[0]
print("Worst = " + str(worst))
median = sorted[len(sorted)//2]
print("Median = " + str(median))
best = sorted[-1]
print("Best = " + str(best))

with open(f"demos/push_worst.pickle", 'wb') as f:
    pickle.dump(ms[worst], f)
with open(f"demos/push_okay.pickle", 'wb') as f:
    pickle.dump(ms[median], f)
with open(f"demos/push_best.pickle", 'wb') as f:
    pickle.dump(ms[best], f)

