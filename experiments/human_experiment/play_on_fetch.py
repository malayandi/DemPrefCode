import sys
sys.path.insert(0,'../..')
import pickle

import numpy as np

import domain
import traj

dom = domain.FetchMove(time_steps=152)
true_weight = [-0.3, -0.3, 0.6, 0.3, -0.7]

def play(name: str):
    states = pickle.load(open(f'generated_demos/{name}.pickle', 'rb'))
    t = traj.Trajectory(np.array([states]), np.array([states]))
    dom.watch(t, on_real_robot=True)

def play_pref(name: str, num: int):
    import pickle
    import domain
    dom = domain.FetchMove(time_steps=152)
    queries = pickle.load(open(f'{name}.pickle', 'rb'))
    t = queries[num]
    dom.watch(t, on_real_robot=True)

if __name__ == "__main__":
    n = sys.argv[1]
    play(n)
