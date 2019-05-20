import time
from typing import Tuple, List
import sys

import numpy as np
import pandas as pd
import pickle

import domain
import human
import query_generation
import sampling
import traj

class Runner(object):
    def __init__(self, domain: domain.Domain, human_type: str,
                 n_demos: int, gen_demos: bool, sim_iter_count: int, demos: List[traj.Trajectory], trim_start: int,
                 n_query: int, update_func: str, query_length: int, inc_prev_query: bool, gen_scenario: bool, n_pref_iters: int, epsilon: float,
                 n_samples_summ: int, n_samples_exp: int,
                 true_weight: List, beta_demo: float, beta_pref: float, beta_human: float):
        """
        Initializes the runner object.

        :param domain: Domain.
        :param human_type: Type of human used; options are "opt", "btz", "term".
        :param n_demos: Number of demos used. If n_demos=0, then not using DemPref.
        :param gen_demos: Whether to generate demos from simulation or use provided demos.
        :param sim_iter_count: Number of iterations to use when generating demos.
        :param demos: Demonstrations provided.
        :param trim_start: Time at which to start trimming the demonstrations.
        :param n_query: Number of queries provided to human at each time step.
        :param update_func: type of update function to use: options are "pick_best", "approx", and "rank".
        :param query_length: Length of each generated query.
        :param inc_prev_query: Whether to include previous-best query in queries presented to human.
        :param gen_scenario: Whether to generate the scenario in each set of queries.
        :param n_pref_iters: Number of iterations to run preference queries for.
        :param epsilon: Minimum distance between the norms of any two queries.
        :param n_samples_summ: Number of samples used to generate summary statistics.
        :param n_samples_exp: Number of samples used to estimate objective in the query generation phase.
        :param true_weight: The true w value; only required if using "opt" or "btz" humans.
        :param beta_demo: Estimated \beta corresponding to demonstrations, to be used in algorithm.
        :param beta_pref: Estimated \beta corresponding to preferences, to be used in algorithm.
        :param human_beta: \beta associated with the human; only used in simulation.
        """
        self.domain = domain
        self.human_type = human_type

        self.n_demos = n_demos
        self.gen_demos = gen_demos
        self.sim_iter_count = sim_iter_count
        if self.n_demos and not self.gen_demos:
            self.demos = demos[:self.n_demos]
        self.trim_start = trim_start

        self.n_query = n_query
        self.update_func = update_func
        self.query_length = query_length
        self.inc_prev_query = inc_prev_query
        self.gen_scenario = gen_scenario
        self.n_pref_iters = n_pref_iters
        self.epsilon = epsilon

        assert self.update_func == "pick_best" or self.update_func == "approx" or self.update_func == "rank", "Update" \
            " function must be one of the provided options"
        if self.inc_prev_query and self.human_type == "term":
            assert self.n_demos > 0, "Cannot include previous query if no demonstration is provided"

        self.n_samples_summ = n_samples_summ
        self.n_samples_exp = n_samples_exp

        self.true_weight = true_weight
        self.beta_demo = beta_demo
        self.beta_pref = beta_pref
        self.beta_human = beta_human

        self.config = [self.human_type, self.n_demos, self.trim_start, self.n_query, self.update_func,
                       self.query_length, self.inc_prev_query, self.gen_scenario, self.n_pref_iters, self.epsilon,
                       self.n_samples_summ, self.n_samples_exp, self.true_weight, self.beta_demo, self.beta_pref,
                       self.beta_human]

    def run(self, n_iters:int=1) -> Tuple[pd.DataFrame, List]:
        """
        Runs the algorithm n_iters times and returns a data frame with all the data from the experiment.

        :param n_iters: Number of times to run the algorithm.
        :param verbose: Prints status messages about the progress of the algorithm if true.
        :return: (self.config, df); config contains the parameters of the run and df is a data frame containing all the
            data from the run.
        """
        ### Creating data frame to store data in
        # run corresponds to the iteration of the whole experiment
        # pref_iter correponds to the iteration of the preference loop in the particular run
        # run is the type of data being stored; options are "mean", "var", "m"
        # value is the actual value being stored
        df = pd.DataFrame(columns=["run #", "pref_iter", "type", "value"])

        ### Creating query generator
        if isinstance(self.domain, domain.Car): # using exact QG when dynamics is available
            if self.update_func == "pick_best":
                obj_fn = query_generation.pick_best
            elif self.update_func == "approx":
                obj_fn = query_generation.approx
            elif self.update_func == "rank":
                obj_fn = query_generation.rank
            qg = query_generation.QueryGenerator(dom=self.domain, num_queries=self.n_query, query_length=self.query_length,
                                                 num_expectation_samples=self.n_samples_exp,
                                                 include_previous_query=self.inc_prev_query,
                                                 generate_scenario=self.gen_scenario,
                                                 objective_fn=obj_fn,
                                                 beta_pref=self.beta_pref)
        else: # using approx QG when dynamics is not available
            qg = query_generation.ApproxQueryGenerator(dom=self.domain, num_queries=self.n_query, query_length=self.query_length,
                                                       num_expectation_samples=self.n_samples_exp,
                                                       include_previous_query=self.inc_prev_query,
                                                       generate_scenario=self.gen_scenario, update_func=self.update_func,
                                                       beta_pref=self.beta_pref)

        ### Creating human
        humans = {
            "opt": human.OptimalHuman(self.domain, self.update_func, self.true_weight),
            "btz": human.BoltzmannHuman(self.domain, self.update_func, self.true_weight, self.beta_human),
            "term": human.TerminalHuman(self.domain, self.update_func)
        }
        H = humans[self.human_type]


        ### Iterating to build confidence intervals
        for i in range(n_iters):
            ### Processing demonstrations
            sampler = sampling.Sampler(n_query=self.n_query, dim_features=self.domain.feature_size,
                                       update_func=self.update_func,
                                       beta_demo=self.beta_demo, beta_pref=self.beta_pref)
            if self.n_demos > 0:
                if self.gen_demos:
                    self.demos = [self.domain.simulate(self.true_weight, iter_count=self.sim_iter_count) for _ in range(self.n_demos)]
                phi_demos = [self.domain.np_features(x) for x in self.demos]
                sampler.load_demo(np.array(phi_demos))
                if self.inc_prev_query and isinstance(self.domain, domain.Car):
                    cleaned_demos = [d.trim(self.query_length, self.trim_start) for d in self.demos]
                else:
                    cleaned_demos = self.demos
                if self.inc_prev_query:
                    last_query_picked = [d for d in cleaned_demos]
            else:
                last_query_picked = [traj.Trajectory(states=None, controls=None, null=True)]

            ## Computing initial estimates
            samples = sampler.sample(N=self.n_samples_summ)
            mean_w = np.mean(samples, axis=0)
            mean_w = mean_w / np.linalg.norm(mean_w)
            var_w = np.var(samples, axis=0)
            data = [[i+1, 0, "mean", mean_w],
                    [i+1, 0, "var", var_w]]
            print("Estimate of w: " + str(mean_w)) # TODO: Add different levels of verbose mode
            print("Estimate of variance: " + str(sum(var_w)))
            # computing convergence measure if we are in simulation
            if self.human_type != "term":
                m = np.mean(
                    [np.dot(w, self.true_weight) / np.linalg.norm(w) / np.linalg.norm(self.true_weight) for w in
                     samples])
                data.append([i+1, 0, "m", m])
                print("Estimate of m: " + str(m) + "\n\n")
            df = df.append(pd.DataFrame(data, columns=["run #", "pref_iter", "type", "value"]), ignore_index=True)

            ### Preferences loop
            for j in range(self.n_pref_iters):
                print("\n\n*** Preferences Loop %d\n" % (j))

                ## Get last_query
                if self.inc_prev_query:
                    if len(self.demos) > 0:
                        random_scenario_index = np.random.randint(len(self.demos))
                    else:
                        random_scenario_index = 0
                    last_query = last_query_picked[random_scenario_index]

                ## Generate queries while ensuring that features of queries are epsilon apart
                query_diff = 0
                print("Generating queries")
                while query_diff <= self.epsilon:
                    if self.inc_prev_query:
                        if last_query.null:
                            queries = qg.queries(samples, blank_traj=True)
                        else:
                            queries = qg.queries(samples, last_query)
                    else:
                        queries = qg.queries(samples)
                    query_diffs = []
                    for m in range(len(queries)):
                        for n in range(m):
                            query_diffs.append(np.linalg.norm(self.domain.np_features(queries[m]) - self.domain.np_features(queries[n])))
                    query_diff = max(query_diffs)

                ## Querying human
                if self.human_type == "term":
                    print('\a')
                rank = H.input(queries)
                if self.update_func == "rank":
                    best = rank[0]
                else:
                    if rank == -1:
                        return df, self.config
                    best = rank

                if self.inc_prev_query:
                    last_query_picked[random_scenario_index] = queries[best]

                ## Creating dictionary mapping rankings to features of queries and loading into sampler
                features = [self.domain.np_features(x) for x in queries]
                phi = {k : features[k] for k in range(len(queries))}
                sampler.load_prefs(phi, rank)

                ## Recording data from this run
                samples = sampler.sample(N=self.n_samples_summ)
                mean_w = np.mean(samples, axis=0)
                mean_w = mean_w / np.linalg.norm(mean_w)
                var_w = np.var(samples, axis=0)
                data = [[i+1, j+1, "mean", mean_w],
                        [i+1, j+1, "var", var_w]]
                print("Estimate of w: " + str(mean_w))
                print("Estimate of variance: " + str(sum(var_w)))
                if self.human_type != "term":
                    m = np.mean(
                        [np.dot(w, self.true_weight) / np.linalg.norm(w) / np.linalg.norm(self.true_weight) for w in
                         samples])
                    data.append([i+1, j+1, "m", m])
                    print("Estimate of m: " + str(m) + "\n\n")
                df = df.append(pd.DataFrame(data, columns=["run #", "pref_iter", "type", "value"]), ignore_index=True)
            ## Resetting for next run
            sampler.clear_pref()
            if self.inc_prev_query and self.n_demos > 0:
                last_query_picked = [d for d in cleaned_demos]

        return df, self.config


if __name__ == "__main__":
    ### COLLECTING AND PROCESSING DEMONSTRATIONS
    dom = domain.Car(dt=0.1, time_steps=50, num_others=1)
    # dom = domain.LunarLander(time_steps=200, seed=0)
    # dom = domain.FetchMove(time_steps=240)

    if isinstance(dom, domain.Car):
        DEMO_WORLD, DEMO_FIXED_CTRL = dom.world_0()
        demo_names = ["andy-a"]
        TRIM_START = 15
        TRUE_WEIGHT = [-0.9, -0.2, 0.9, 0.05]
    else:
        demo_names = ["pipelines/fetch/gleb_demo_1"]
        TRIM_START = 0
        TRUE_WEIGHT = [-0.9, -0.2, 0.9, 0.05]
    TRUE_WEIGHT /= np.linalg.norm(TRUE_WEIGHT)

    ### SETTING PARAMETERS OF ALGORITHM
    HUMAN_TYPE = "opt"

    N_DEMOS = 1
    GEN_DEMOS = True
    SIM_ITER_COUNT = 100
    demos = []
    if N_DEMOS and not GEN_DEMOS:
        for d in demo_names:
            demo = pickle.load(open(str(d) + ".pickle", 'rb'), encoding='bytes')
            demos.append(demo)

    N_QUERY = 2
    UPDATE_FUNC = "pick_best"
    QUERY_LENGTH = 3
    INC_PREV_QUERY = False
    if isinstance(dom, domain.Car):
        GEN_SCENARIO = True
    else:
        GEN_SCENARIO = False
    N_PREF_ITERS = 25
    EPSILON = 0

    N_SAMPLES_SUMM = 50000
    N_SAMPLES_EXP = N_SAMPLES_SUMM

    BETA_DEMO = 0.1
    BETA_PREF = 5
    BETA_HUMAN = 1

    ### STANDARD RUNNER CALL
    runner = Runner(dom, HUMAN_TYPE,
                    N_DEMOS, GEN_DEMOS, SIM_ITER_COUNT, demos, TRIM_START,
                    N_QUERY, UPDATE_FUNC, QUERY_LENGTH, INC_PREV_QUERY, GEN_SCENARIO, N_PREF_ITERS, EPSILON,
                    N_SAMPLES_SUMM, N_SAMPLES_EXP,
                    TRUE_WEIGHT, BETA_DEMO, BETA_PREF, BETA_HUMAN)
    df, config = runner.run(n_iters=1)

    # name = "exp_results/dempref=" + str(DEMPREF) + ",b_demo,b_pref=" + str(BETA_DEMO) + "," + str(BETA_PREF)
    name = "fetch_exp" + str(time.time())
    with open(name + "_db.pickle", 'wb') as f:
        pickle.dump(df, f)
    with open(name + "_config.pickle", 'wb') as f:
        pickle.dump(config, f)

