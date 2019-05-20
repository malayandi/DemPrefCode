import pickle, time
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import domain
#from gym.envs.classic_control import rendering # DO NOT DELETE
import dist_plot
import sampling
import traj

class Human(object):
    """
        Use Human as an interface for obtaining preferences over histories.

        Human is an abstract class, and must be subclassed and implemented.

        >>> class MyHuman(object):
        ...   def __init__(self):
        ...     # implementation
        ...   def input(self) -> int:
        ...     # implementation
        ...
        >>> h = MyHuman()
        >>> selected_index = h.input(histories)
        >>> def function_that_takes_human(h: human.Human):
        ...   # This function can expect h.input to be defined, 
        ...   # but doesn't rely on a particular implementation.
    """

    def __init__(self, domain: domain.Domain, type="select"):
        """
        Generic init method.

        :param domain: the domain in which the Human is responding.
        :param type: the type of the update function used -- use "select" if the update function is "pick_best" or
            "approx" and use "rank" if the update function is "rank".
        """
        self.domain = domain
        self.type = type # options are select and rank

    def input(self, queries: List[traj.Trajectory]) -> int:
        """
            Use input to ask for the trajactory preferences. 
            See concrete implementations for more information.
        """
        raise NotImplementedError("Human.input")

Terminal_TRAJECTORY_PROMPT          = "Trajectory #"
Terminal_SELECTED_TRAJECTORY_PROMPT = "Your Selected Trajectory # [1,...,%d]:"
Terminal_RANKING_PROMPT = "Your Ranking of the Trajectories [1,...,%d]: (e.g. (2, 3, 1) means Trajectory 3 is your most" \
                          " preferred, followed by Trajectory 1, and then lastly, 2.)\n"

class TerminalHuman(Human):
    """
        TerminalHuman implements the interface by a human can, via the command line (terminal)
        view and select trajectories.
    """
    def __init__(self, domain: domain.Domain, type: str):
        """
        Initializes the TerminalHuman.

        :param domain: see abstract class documentation.
        :param type: see abstract class documentation.
        """
        self.domain = domain
        self.type = type
        assert self.type == "pick_best" or self.type == "approx" or self.type == "rank"

    def input(self, queries, on_real_robot=False):
        print("To view a trajectory, press a number: [1,...,%d]; when ready to select input 'done'" % (len(queries)))
        inp = ''
        while inp != 'done':
            inp = input(Terminal_TRAJECTORY_PROMPT)
            if inp == "save":
                with open(str(time.time()) + ".pickle", 'wb') as f:
                    pickle.dump(queries, f)
            if inp == 'debug':
                import pdb; pdb.set_trace()
                continue
            try:
                index = int(inp) # This statement can fail, reaching except clause.

                if index < 1 or index > len(queries):
                    print("Please input a number in the proper range")
                    continue
            except:
                print("Please input a number")
                continue

            traj = queries[index - 1]

            if isinstance(self.domain, domain.Car):
                self.domain.w.initial_state = [
                    traj.states[0, 0, :],
                    traj.states[0, 1, :],
                ]
            self.domain.reset()
            if isinstance(self.domain, domain.FetchMove):
                self.domain.watch(traj, on_real_robot)
            self.domain.reset()
        print('\n')

        while True:
            if self.type != "rank":
                inp = input(Terminal_SELECTED_TRAJECTORY_PROMPT % len(queries))
                try:
                    sample = int(inp)
                    return sample - 1 # we allow user to think starting from 1
                except:
                    print("Please input a number")
                    continue
            elif self.type == "rank":
                inp = input(Terminal_RANKING_PROMPT % len(queries))
                try:
                    inp = list(inp.replace(",", "").replace(" ", ""))[1:-1]
                    inp = [int(i) - 1 for i in inp]
                    inp = tuple(inp)
                    return inp
                except:
                    print("Please input a valid ranking")
                    continue



class OptimalHuman(Human):
    """
        OptimalHuman always selects the history that maximizes rewards according to self.true_weights
    """

    def __init__(self, domain: domain.Domain, type: str, weights: np.ndarray):
        """
        Initializes the OptimalHuman.

        :param domain: see abstract class documentation.
        :param type: see abstract class documentation.
        :param weights: the true reward weights.
        """
        self.true_weights = np.asarray(weights) # the real reward weights.
        self.domain = domain
        self.type = type
        assert self.type == "pick_best" or self.type == "approx" or self.type == "rank"

    def input(self, queries):
        features = [self.domain.np_features(x) for x in queries]
        rewards = [self.true_weights.dot(f) for f in features]
        if self.type != "rank":
            return np.argmax(rewards)
        else:
            sorted = np.argsort(rewards[:])
            sorted = sorted[::-1]
            return tuple(sorted)


class BoltzmannHuman(Human):
    """
        BoltzmannHuman simulates _random_ input from a human behaving according
        to the Boltzmann model (softmax).

        The probability that BoltzmannHuman selects a history is proportional to
        exp(beta * reward according to self.true_weights).
    """
    def __init__(self, domain: domain.Domain, type: str, weights: np.ndarray, beta:float=1.0):
        """
        Initializes the BoltzmannHuman.

        :param domain: see abstract class documentation.
        :param type: see abstract class documentation.
        :param weights: the true reward weights.
        :param beta: the "rationality" parameter.
        """
        self.true_weights = weights
        self.domain = domain
        self.beta = beta
        self.type = type
        assert self.type == "pick_best" or self.type == "approx" or self.type == "rank"

    def input(self, queries):
        features = [self.domain.np_features(x) for x in queries]
        rewards = [np.exp(self.beta * self.true_weights.dot(f)) for f in features]
        probs = rewards / np.sum(rewards)
        if self.type != "rank":
            return np.random.choice(len(features), p=probs)
        else:
            result = []
            while rewards:
                index = np.random.choice(len(features), p=probs)
                result.append(index)
                rewards.pop(index)
            return tuple(result)

