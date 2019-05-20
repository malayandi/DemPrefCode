import sys
sys.path.insert(0,'../..')

import pickle
import typing

import numpy as np

import domain
import runner

HUMAN_TYPE = "opt"

N_DEMOS = 1
GEN_DEMOS = False
SIM_ITER_COUNT = 100
QUERY_LENGTH = 10
N_PREF_ITERS = 25
N_ITERS_EXP = 8

N_QUERY = 2
UPDATE_FUNC = "rank"
EPSILON = 0
N_SAMPLES_SUMM = 50000
N_SAMPLES_EXP = N_SAMPLES_SUMM

BETA_DEMO = 0.1
BETA_PREF = 5
BETA_HUMAN = 1

def main_experiment(dom: domain.Domain, true_weight: typing.List, name: str):
    """
    Runs a full factorial experiment on {dempref, non-dempref} and {CIA, non-CIA} in simulation. Saves the data to a
    file locally.

    :param domain: the domain on which to run the experiment.
    :param true_weight: reward weight vector for the simulated agent to optimize.
    :param name: name of domain, as a string.
    """
    if isinstance(dom, domain.Car):
        trim_start = 15
        gen_scenario = True
    else:
        trim_start = 0
        gen_scenario = False

    ### STANDARD RUNNER CALL
    errors = []
    for quality in ["worst", "best"]:
        demos = [pickle.load(open(f"demos/{name}_{quality}.pickle", 'rb'), encoding='bytes')]
        for inc_prev_query in [True, False]:
            print("\n===")
            print(f"quality = {quality}, IC = {inc_prev_query}")
            print("===\n")
            n_query = N_QUERY + 1 if inc_prev_query else N_QUERY
            r = runner.Runner(dom, HUMAN_TYPE,
                              N_DEMOS, GEN_DEMOS, SIM_ITER_COUNT, demos, trim_start,
                              n_query, UPDATE_FUNC, QUERY_LENGTH, inc_prev_query, gen_scenario, N_PREF_ITERS, EPSILON,
                              N_SAMPLES_SUMM, N_SAMPLES_EXP,
                              true_weight, BETA_DEMO, BETA_PREF, BETA_HUMAN)
            n = f"results/domain={name},quality={quality},IC={inc_prev_query}"
            try:
                df, config = r.run(n_iters=N_ITERS_EXP)
            except:
                errors.append(n)

            with open(n + "_db.pickle", 'wb') as f:
                pickle.dump(df, f)
            with open(n + "_config.pickle", 'wb') as f:
                pickle.dump(config, f)
            if errors:
                with open(n + "_errors.pickle", 'wb') as f:
                    pickle.dump(errors, f)


if __name__ == "__main__":
    args = sys.argv
    D = args[1] # options are "driver", "lander", "fetch_move"

    if D == "driver":
        DOM = domain.Car(dt=0.1, time_steps=50, num_others=1)
        TRUE_WEIGHT = [0.5, -0.2, 0.2, -0.7]
    elif D == "lander":
        DOM = domain.LunarLander(time_steps=150)
        TRUE_WEIGHT = [-0.4, 0.4, -0.2, -0.7]
    else:
        print(f"No domain named {D}.")

    main_experiment(DOM, TRUE_WEIGHT, D)
