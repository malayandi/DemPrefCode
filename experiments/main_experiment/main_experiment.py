import sys
sys.path.insert(0,'../..')

import pickle
import typing

import numpy as np

import domain
import runner

HUMAN_TYPE = "opt"

SIM_ITER_COUNT = 100
N_PREF_ITERS = 25
N_ITERS_EXP = 8

N_QUERY = 2
INC_PREV_QUERY = False
UPDATE_FUNC = "approx"
EPSILON = 0
N_SAMPLES_SUMM = 50000
N_SAMPLES_EXP = N_SAMPLES_SUMM

BETA_DEMO = 0.1
BETA_PREF = 5
BETA_HUMAN = 1

def main_experiment(dom: domain.Domain, true_weight: typing.List, gen_demos: bool, demos: typing.List, query_length: int, name: str):
    """
    Runs a full factorial experiment on {dempref, non-dempref} and {CIA, non-CIA} in simulation. Saves the data to a
    file locally.

    :param dom: the domain on which to run the experiment.
    :param true_weight: reward weight vector for the simulated agent to optimize.
    :param gen_demos: whether to generate demos or use provided demos.
    :param demos: demonstrations provided.
    :param query_length: length of query length.
    :param name: name of domain, as a string.
    :return:
    """
    if isinstance(dom, domain.Car):
        trim_start = 15
        gen_scenario = True
    else:
        trim_start = 0
        gen_scenario = False

    ### STANDARD RUNNER CALL
    errors = []
    for n_demos in [0, 1, 3]:
        print("\n===")
        print(f"n_demos={n_demos}")
        print("===\n")
        r = runner.Runner(dom, HUMAN_TYPE,
                          n_demos, gen_demos, SIM_ITER_COUNT, demos, trim_start,
                          N_QUERY, UPDATE_FUNC, query_length, INC_PREV_QUERY, gen_scenario, N_PREF_ITERS, EPSILON,
                          N_SAMPLES_SUMM, N_SAMPLES_EXP,
                          true_weight, BETA_DEMO, BETA_PREF, BETA_HUMAN)
        n = f"results/domain={name},n_demos={n_demos}"
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
        GEN_DEMOS = True
        DEMOS = []
        QUERY_LENGTH = 10
    elif D == "lander":
        DOM = domain.LunarLander(time_steps=150)
        TRUE_WEIGHT = [-0.4, 0.4, -0.2, -0.7]
        GEN_DEMOS = True
        DEMOS = []
        QUERY_LENGTH = 10
    elif D == "fetch_move":
        DOM = domain.FetchMove(time_steps=150)
        TRUE_WEIGHT = [-0.6, -0.3, 0.9]
        GEN_DEMOS = False
        DEMOS = [pickle.load(open(f"fetch_demos/{D}.pickle", 'rb'), encoding='bytes')]
        QUERY_LENGTH = 5
    else:
        print(f"No domain named {D}.")

    main_experiment(DOM, TRUE_WEIGHT, GEN_DEMOS, DEMOS, QUERY_LENGTH, D)
