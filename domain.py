import copy
import math
import time
import typing

import gym
import numpy as np
import scipy.optimize as opt
import theano as th
import theano.tensor as tt

import carDomain.car as car
import carDomain.dynamics as dynamics
import carDomain.lane as lane
import carDomain.world as world
import carDomain.visualize as visualize

import gym.keyboard_agent as keyboard_agent
import gym.envs.robotics.utils as fetch_utils

import traj as traj
import utils

import uuid
import pickle
import sys


BoundsType = typing.List[typing.Tuple[float, float]]
FeaturesFuncType = typing.Callable[
    [tt.TensorVariable, tt.TensorVariable],
    tt.TensorVariable,
]
DynamicsFuncType = typing.Callable[
    [tt.TensorVariable, tt.TensorVariable],
    tt.TensorVariable,
]


class Domain(object):
    def __init__(self,
                 state_size: int,             # Size of the state space.
                 state_bounds: BoundsType,    # Bounds on the state variables.
                 control_size: int,           # Size of the control space.
                 control_bounds: BoundsType,  # Bounds on control variables.
                 feature_size: int,           # Size of the feature space.
                 features: FeaturesFuncType,  # Features p(x, other_xs).
                 dynamics: DynamicsFuncType,  # Dynamics f(x, u).
                 time_steps: int,             # Number time steps for episode.
                 num_others: int,             # Number other agents. May be 0.
                 ) -> None:
        """
        Initializes the domain object.

        :param state_size: dimensionality of state space
        :param state_bounds: a list of tuples, containing the min and max value of each dimension of the state space
        :param control_size: dimensionality of control space
        :param control_bounds: a list of tuples, containing the min and max value of each dimension of the control space
        :param feature_size: dimensionality of the feature space
        :param features: theano function that returns the features for a given state
        :param dynamics: function that returns the next state given the current state and control
        :param time_steps: length of each trajectory
        :param num_others: number of other agents in the domain
        """
        self.state_size = state_size
        self.state_bounds = state_bounds
        self.control_size = control_size
        self.control_bounds = control_bounds
        self.feature_size = feature_size
        self.features_function = features
        self.dynamics_function = dynamics
        self.time_steps = time_steps
        self.num_others = num_others
        self.num_agents = num_others + 1

    def reset(self):
        """
        Resets the domain to its original state.

        :return:
        """
        raise NotImplementedError()

    def run(self, controls: np.ndarray) -> traj.Trajectory:
        """
        Simulates the entire environment for the given controls as is and returns the generated trajectory.

        :return:
        """
        raise NotImplementedError()


    def watch(self, t: traj.Trajectory) -> None:
        """
        Plays the given trajectory.

        :param t: trajectory to be played.
        :param on_real_robot: whether to watch the trajectory in simulation or on the real robot. Only needed if using
            Fetch domain.
        :return:
        """
        raise NotImplementedError()

    def simulate(self, weights: np.ndarray) -> traj.Trajectory:
        """
        Given the weights of the reward function, simulate the behavior of an agent behaving according to the reward
        function.

        :param weights:
        :return:
        """
        raise NotImplementedError()

    def collect_dems(self, num:int=1) -> typing.List[str]:
        """
        Collect demonstrations.

        :param num: Number of demonstrations to be collected.
        :return:
        """
        raise NotImplementedError()

    def tt_features(self, x: np.ndarray, other_xs: np.ndarray):
        """
        Theano function to compute features when the main agent is at state x and the other agents in the system are at
        states other_xs.

        ONLY REQUIRED IF DIFFERENTIATING THROUGH THE COMPUTATION GRAPH. (i.e., not needed if using
        ApproxQueryGenerator.)

        :param x: state of main agent.
        :param other_xs: list containing states of other agents.
        :return:
        """
        raise NotImplementedError()

    def np_features(self, t: traj.Trajectory):
        """
        Computes the function across a whole trajectory and returns a Numpy list containing the features.

        :param t:
        :return:
        """
        raise NotImplementedError()

class Gym(Domain):
    def __init__(self, name: str, num_features: int=4, seed: int=0, time_steps: int=250, frame_delay_ms: int=20,
                 state_size: int=None, state_bounds: typing.List=None, control_size: int=None, control_bounds: typing.List=None):
        """
        Initializes the Gym domain.
        :param name: Name of the environment to make.
        :param num_features: Number of features.
        :param seed: Seed that will be used to generate the world for demonstrations and preferences.
        :param time_steps: Length of trajectory to be watched.
        :param frame_delay_ms: Delay for animation.
        :param state_size: Needed for Fetch integration. For non-Fetch environments, leave blank.
        :param state_bounds: Needed for Fetch integration. For non-Fetch environments, leave blank.
        :param control_size: Needed for Fetch integration. For non-Fetch environments, leave blank.
        :param control_bounds: Needed for Fetch integration. For non-Fetch environments, leave blank.
        """
        self.env = gym.make(name)
        self.seed = seed
        self.env.seed(self.seed)

        self.frame_delay_ms= frame_delay_ms

        state_size = state_size if state_size else self.env.observation_space.shape[0]
        state_bounds = state_bounds if state_bounds else [(self.env.observation_space.low[i], self.env.observation_space.high[i]) for i in range(self.env.observation_space.shape[0])]
        control_size = control_size if control_size else self.env.action_space.shape[0]
        control_bounds = control_bounds if control_bounds else [(self.env.action_space.low[i], self.env.action_space.high[i]) for i in range(self.env.action_space.shape[0])]

        super(Gym, self).__init__(state_size=state_size,
                                  state_bounds=state_bounds,
                                  control_size=control_size,
                                  control_bounds=control_bounds,
                                  feature_size=num_features,
                                  features=None,
                                  dynamics=None,
                                  time_steps=time_steps,
                                  num_others=0)

    def reset(self, seed:int=None):
        if not seed:
            seed = self.seed
        self.env.seed(seed)
        state = self.env.reset()
        return state

    def run(self, controls: np.ndarray) -> traj.Trajectory:
        c = np.array([[0.] * self.control_size] * self.time_steps)
        num_intervals = len(controls)//self.control_size
        interval_length = self.time_steps//num_intervals

        assert interval_length * num_intervals == self.time_steps, "Number of generated controls must be divisible by total time steps."

        j = 0
        for i in range(num_intervals):
            c[i * interval_length: (i + 1) * interval_length] = [controls[j + i] for i in range(self.control_size)]
            j += self.control_size

        obser = self.reset()
        s = [obser]
        for i in range(self.time_steps):
            try:
                results = self.env.step(c[i])
            except:
                print("Caught unstable simulation; skipping.")
                return traj.Trajectory(None, None, null=True)
            if isinstance(self, Fetch):
                obser = self.state
            else:
                obser = results[0]
            s.append(obser)
            if results[2]:
                break
        if len(s) <= self.time_steps:
            c = c[:len(s), :]
        else:
            c = np.append(c, [np.zeros(self.control_size)], axis=0)
        return traj.Trajectory(np.array([s]), np.array([c]))

    # def watch(self, t: traj.Trajectory, seed:int=None, on_real_robot:bool=False):
    #     self.reset(seed)
    #     self.env.render()
    #     for i in range(len(t.controls[0])):
    #         a = t.controls[0][i]
    #         results = self.env.step(a)
    #         #print(self.state)
    #         time.sleep(self.frame_delay_ms/1000)
    #         if results[2]: # quit if game over
    #             break
    #     self.env.close()
    #     self.reset()

    def simulate(self, weights: np.ndarray, seed=None, query_length: int=10, iter_count=1) -> traj.Trajectory:
        def reward(controls, domain, weights):
            """
            One-step reward function.

            :param x: state
            :param weights: weight for reward function
            :return:
            """
            t = self.run(controls)
            return -np.dot(weights, domain.np_features(t)) # TODO: Maybe add incentive to minimize control size

        weights = weights/np.linalg.norm(weights)
        self.reset(seed)

        low = [x[0] for x in self.control_bounds] * query_length
        high = [x[1] for x in self.control_bounds] * query_length
        optimal_ctrl = None
        opt_val = np.inf
        start = time.time()
        for _ in range(iter_count):
            temp_res = opt.fmin_l_bfgs_b(reward, x0=np.random.uniform(low=low, high=high, size=self.control_size * query_length),
                                         args=(self, weights), bounds=self.control_bounds * query_length,
                                         approx_grad=True, maxfun=1000, maxiter=100)
            if temp_res[1] < opt_val:
                optimal_ctrl = temp_res[0]
                opt_val = temp_res[1]
        end = time.time()
        print("Finished generating simulated behavior in " + str(end-start) + "s")

        t = self.run(optimal_ctrl)
        self.reset()
        return t


class Fetch(Gym):
    state_size = 10
    control_size = 7

    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint",
                     "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
    joint_ranges = [(-1.6056, 1.6056), (-1.221, 1.518), (-np.pi, np.pi), (-2.251, 2.251),
                    (-np.pi, np.pi), (-2.16, 2.16), (-np.pi, np.pi), (-10, 10), (-10, 10), (-10, 10)]

    control_ranges = [(-1,1 ), (-1, 1), (-1, 1), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]


    def __init__(self, name: str, seed: int=0, num_features: int=4, time_steps: int=50):
        super(Fetch, self).__init__(name=name, num_features=num_features, seed=seed, time_steps=time_steps,
                                    state_size=self.state_size, state_bounds=self.joint_ranges,
                                    control_size=self.control_size, control_bounds=self.control_ranges)
        self.env.unwrapped.sim.data.set_joint_qpos("robot0:torso_lift_joint", 0.38)
        # print(fetch_utils.robot_get_obs(self.env.unwrapped.sim)[0]) # Full vectors of joint angles and velocities

    def reset(self, seed: int=None):
        super(Fetch, self).reset(seed)
        self.env.unwrapped.sim.data.set_joint_qpos("robot0:torso_lift_joint", 0.38)
        return self.state

    @property
    def state(self):
        s = []
        for name in self.joint_names:
            s.append(self.env.unwrapped.sim.data.get_joint_qpos("robot0:" + name))
        s.extend(list(self.env.unwrapped.sim.data.get_site_xpos("robot0:grip")))
        return s

    @state.setter
    def state(self, s):
        for i in range(len(self.joint_names)):
            self.env.unwrapped.sim.data.set_joint_qpos("robot0:" + self.joint_names[i], s[i])

    def fetch_to_mujoco(self, t: traj.Trajectory) -> traj.Trajectory:
        self.reset()
        new_states = []
        for i in range(len(t.states[0][1:])):
            self.state = t.states[0][i]
            results = self.env.step(t.controls[0][i-1])
            self.state.extend(results[:3])
            new_states.append(self.state)
        self.reset()
        return traj.Trajectory(np.array([new_states]), np.array([t.controls[0][1:]]))

    def mujoco_to_fetch(self, t: traj.Trajectory) -> traj.Trajectory:
        new_states = []
        for x in t.states[0]:
            new_states.append(x[:7])
        return traj.Trajectory(np.array([new_states]), t.controls)

    def collect_dems(self, world_lst: typing.List=None) -> typing.List[str]:
        return record_runner(self.time_steps, 0.5)

    def watch(self, t: traj.Trajectory, on_real_robot:bool=False):
        """
        Does the same thing as Gym's Fetch call except adds an option to watch the trajectory on the robot instead of
        simulation.

        :param t: trajectory to be played.
        :param on_real_robot: boolean indicating whether to play the trajectory on the real robot or the simulation,
        :return: 
        """
        if not on_real_robot:
            self.reset()
            self.env.render()
            for i in range(len(t.states[0])):
                s = t.states[0][i]
                self.state = s
                self.env.render()
                # print(self.state)
                time.sleep(self.frame_delay_ms/1000)
            self.env.close()
            self.env.env.close()
            self.reset()
        else:
            traj_file_string = str(uuid.uuid4()) + '.pickle'
            with open('dempref_demonstrations/' + traj_file_string, 'wb') as traj_file:
                pickle.dump(t, traj_file, protocol=2)
                playback_runner(traj_file_string)


class FetchMove(Fetch):
    start_state = [1.364, -0.294, -2.948, 0.906, -0.275, -1.206, 3.086]

    def __init__(self, seed: int=0, time_steps: int=50, test: bool=False):
        n = "FetchReach-v1" if not test else "FetchReachTest-v1"
        if test:
            self.start_state[0] = -self.start_state[0]
        super(FetchMove, self).__init__(name=n, seed=seed, num_features=3, time_steps=time_steps)
        self.state = self.start_state
        self.table_pos = self.env.unwrapped.sim.data.get_body_xpos('table')
        self.obstacle_pos = self.env.unwrapped.sim.data.get_body_xpos('boxobstacle')
        self.goal_pos = self.env.unwrapped.sim.data.get_body_xpos('goal')
        self.initial_ncon = None

    def reset(self, seed: int=None):
        super(Fetch, self).reset(seed)
    
        self.env.unwrapped.sim.data.set_joint_qpos("robot0:torso_lift_joint", 0.38)
        self.state = self.start_state
        self.initial_ncon = int(self.env.unwrapped.sim.data.ncon)
        return self.state

    def np_features(self, t: traj.Trajectory):
        #if t.null:
        #    return np.array([-np.inf] * self.feature_size)
        def np_goal_distance(x):
            goal_pos = self.goal_pos
            return 25 * -np.exp(-np.sqrt((x[7] - goal_pos[0]) ** 2 + (x[8] - goal_pos[1]) ** 2 + (x[9] - goal_pos[2]) ** 2))

        def np_table_distance(x):
            table_pos = self.table_pos
            return 5 * -np.exp(-np.sqrt((x[9] - table_pos[2]) ** 2))

        def np_obstacle_distance(x):
            obstacle_pos = self.obstacle_pos
            return 40 * (1 - np.exp(-np.sqrt((x[7] - obstacle_pos[0]) ** 2 + (x[8] - obstacle_pos[1]) ** 2 + (x[9] - obstacle_pos[2]) ** 2)))

        lst_of_features = []
        for x in t.states[0]:
            phi = np.stack([
                np_goal_distance(x),
                np_table_distance(x),
                np_obstacle_distance(x),
            ])
            lst_of_features.append(phi)

        phi_total = list(np.mean(lst_of_features, axis=0))
        return np.array(phi_total)


class LunarLander(Gym):
    def __init__(self, seed: int=77, time_steps: int=250):
        super(LunarLander, self).__init__(name="LunarLanderContinuous-v2", num_features=4, seed=seed, time_steps=time_steps)

    def watch(self, t: traj.Trajectory, seed: int = None):
        if len(t.controls[0][0]) == 1:
            mapping = {1: [0, -1], 2: [1, 0], 3: [0, 1], 0: [0, 0]}
            controls = []
            for i in range(len(t.controls[0])):
                controls.append(mapping[t.controls[0][i][0]])
            t = traj.Trajectory(t.states, np.array([controls]))
        super(LunarLander, self).watch(t, seed)

    def collect_dems(self, num_dems: int=1, world_lst: typing.List=None) -> typing.List[str]:
        names = []
        for _ in range(num_dems):
            name = keyboard_agent.play(self.seed)
            names.append(name)
        return names

    def np_features(self, t: traj.Trajectory):
        # distance from landing pad at (0, 0)
        # weight should be negative
        def np_dist_from_landing_pad(x):
            return -15*np.exp(-np.sqrt(x[0]**2+x[1]**2))

        # angle of lander
        # angle is 0 when upright (positive in left direction, negative in right)
        # weight should be positive
        def np_lander_angle(x):
            return 15*np.exp(-np.abs(x[4]))

        # velocity of lander
        # weight should be negative
        def np_velocity(x):
            return -10*np.exp(-np.sqrt(x[2]**2+x[3]**2))

        # total path length
        # weight should be positive
        def np_path_length(t):
            states = t.states[0]
            total = 0
            for i in range(1, len(states)):
                total += np.sqrt((states[i][0] - states[i-1][0])**2 + (states[i][1] - states[i-1][1])**2)
            total = np.exp(-total)
            return 10 * total
        
        # final position
        # weight should be negative
        def np_final_position(t):
            x = t.states[0][-1]
            return -30*np.exp(-np.sqrt(x[0] ** 2 + x[1] ** 2))

        lst_of_features = []
        for i in range(len(t.states[0])):
            x = t.states[0][i]
            if i > len(t.states)//5:
                phi = np.stack([
                    np_dist_from_landing_pad(x),
                    np_lander_angle(x),
                    np_velocity(x),
                ])
            else:
                phi = np.stack([
                    np_dist_from_landing_pad(x),
                    np_lander_angle(x),
                    0,
                ])
            lst_of_features.append(phi)
        phi_total = list(np.mean(lst_of_features, axis=0))
        # phi_total.append(np_path_length(t))
        phi_total.append(np_final_position(t))
        return np.array(phi_total)

class Car(Domain):
    car_control_bounds = [(-1., 1.), (-1., 1.)]
    car_state_bounds = [
        (-0.15, 0.15),
        (-0.1, 0.2),
        (math.pi*0.4, math.pi*0.6),
        (0., 1.),
    ]

    def __init__(self, dt: float, time_steps: int, num_others: int) -> None:
        super(Car, self).__init__(state_size=4,
                                  state_bounds=self.car_state_bounds,
                                  control_size=2,
                                  control_bounds=self.car_control_bounds,
                                  feature_size=4,
                                  features=self.tt_features,
                                  dynamics=dynamics.CarDynamics(dt=dt),
                                  time_steps=time_steps,
                                  num_others=num_others,
                                  )
        self.dt = dt

        self.w, self.fixed_ctrl = self.world_0()

    def world_0(self) -> (world.World, typing.List):
        """
        Returns the world where the human car merges into the robot car's lane, along with the fixed control for the
        other car.

        :return:
        """
        w = world.World()
        clane = lane.StraightLane([0., -1.], [0., 1.], 0.17)
        w.lanes += [clane, clane.shifted(1), clane.shifted(-1)] # 3 lanes
        w.roads += [clane]
        w.fences += [clane.shifted(2), clane.shifted(-2)]

        robot = car.Car(self.dynamics_function, [0., -0.4, np.pi/2., 0.4], color='orange') # 1 robot car -- the optimizer
        human = car.Car(self.dynamics_function, [0.17, -0., np.pi/2., 0.4], color='white') # 1 human car -- with fixed behavior (see run())
        w.initial_state = [robot.x, human.x]

        w.cars.append(robot)
        w.cars.append(human)

        fixed_ctrl = []
        for i in range(self.time_steps):
            if i < self.time_steps // 5:
                ctrl = [0, w.cars[1].traj.x0[3].eval()]
            elif i < 2 * self.time_steps // 5:
                ctrl = [1., w.cars[1].traj.x0[3].eval()]
            elif i < 3 * self.time_steps // 5:
                ctrl = [-1., w.cars[1].traj.x0[3].eval()]
            elif i < 4 * self.time_steps // 5:
                ctrl = [0, w.cars[1].traj.x0[3].eval() * 1.3]
            else:
                ctrl = [0, w.cars[1].traj.x0[3].eval() * 1.3]
            fixed_ctrl.append(ctrl)

        return w, fixed_ctrl

    def world_1(self) -> (world.World, typing.List):
        """
        Returns the world where the human car slows down into the robot car, along with the fixed control for the
        other car.

        :return:
        """
        w = world.World()
        clane = lane.StraightLane([0., -1.], [0., 1.], 0.17)
        w.lanes += [clane, clane.shifted(1), clane.shifted(-1)] # 3 lanes
        w.roads += [clane]
        w.fences += [clane.shifted(2), clane.shifted(-2)]

        robot = car.Car(self.dynamics_function, [0., -0.3, np.pi/2., 0.4], color='orange') # 1 robot car -- the optimizer
        human = car.Car(self.dynamics_function, [0, 0., np.pi/2., 0.4], color='white') # 1 human car -- with fixed behavior (see run())
        w.initial_state = [robot.x, human.x]

        w.cars.append(robot)
        w.cars.append(human)

        fixed_ctrl = []
        for i in range(self.time_steps):
            if i < self.time_steps // 5:
                ctrl = [0, w.cars[1].traj.x0[3].eval()]
            elif i < 2 * self.time_steps // 5:
                ctrl = [0, 0]
            elif i < 3 * self.time_steps // 5:
                ctrl = [0, 0]
            elif i < 4 * self.time_steps // 5:
                ctrl = [0, w.cars[1].traj.x0[3].eval()]
            else:
                ctrl = [0, w.cars[1].traj.x0[3].eval()]
            fixed_ctrl.append(ctrl)

        return w, fixed_ctrl


    def reset(self, w: world.World=None):
        """
        Resets the provided world w. If no w is provided, self.w is reset.

        :return:
        """
        if not w:
            w = self.w
        w.cars[0].x = w.initial_state[0]
        w.cars[1].x = w.initial_state[1]
        w.cars[0].data0['x0'] = w.initial_state[0]
        w.cars[1].data0['x0'] = w.initial_state[1]

    def watch(self, t: traj.Trajectory, w: world.World=None):
        """
        Replays the given trajectory in the specified world; if no world is given, uses the world with which the domain
        was initialized.

        :param t: trajectory to watch.
        :param w: world object on which trajectory was collected.
        :param repeat_count: number of times to repeat the viewing.
        :return:
        """
        if not w:
            w = self.w
        # w.cars[0].x = w.initial_state[0]
        # w.cars[1].x = w.initial_state[1]
        w.cars[0].x = t.states[0][0]
        w.cars[1].x = t.states[1][0]
        w.cars[0].data0['x0'] = t.states[0][0]
        w.cars[1].data0['x0'] = t.states[1][0]


        viewer = visualize.Visualizer(self.dt, magnify=1.2)
        viewer.main_car = w.cars[0]
        viewer.use_world(w)
        viewer.paused = True
        viewer.run_modified(history_x=t.states, history_u=t.controls)
        viewer.window.close()
        self.reset(w)


    def simulate(self, weights: np.ndarray, w: world.World=None, fixed_ctrl: typing.List=None, iter_count:int =0, random_start:bool =False) -> traj.Trajectory:
        """
        Simulates the behavior generated by a simpleOptimizerCar behaving according to w_true in the given world. If no
        world or fixed_ctrl is provided, will run with self.w and self.fixed_ctrl.

        :param weights: the true weight of the reward function according to which the car will behave.
        :param w: the world on which to simulate the car's behavior.
        :param fixed_ctrl: the fixed behavior of the other car in the world.
        :return:
        """
        if not w:
            w = self.w
        if not random_start:
            w.cars[0] = car.SimpleOptimizerCar(self.dynamics_function, w.initial_state[0], color='orange')
        else:
            state = w.initial_state[0]
            state[0] = np.random.uniform(-1, 1)
            state[1] = np.random.uniform(-0.5, 0.5)
            w.cars[0] = car.SimpleOptimizerCar(self.dynamics_function, state, color='orange')
        w.cars[1] = car.UserControlledCar(self.dynamics_function, w.initial_state[1], color='white')
        w.cars[1].fix_control(fixed_ctrl)

        def reward(weights):
            def f(t, x, u):
                return tt.dot(self.tt_features(x, [w.cars[1].x]), weights) - 0.01 * (u[0] ** 2 + u[1] ** 2)
            return f

        weights = weights/np.linalg.norm(weights)
        w.cars[0].reward = reward(weights)
        states = [[] for _ in w.cars]
        controls = [[] for _ in w.cars]

        for i in range(self.time_steps):
            w.cars[0].control(0, 0)
            w.cars[1].control(0, 0)
            for c, hist in zip(w.cars, controls):
                hist.append(c.u)
            for c in w.cars:
                c.move()
            for c, hist in zip(w.cars, states):
                hist.append(c.x)

        t = traj.Trajectory(np.array(states), np.array(controls))
        return t

    def collect_dems(self, num:int=1, world_lst:typing.List[world.World]=None, fixed_ctrl_lst:typing.List=None) -> typing.List[str]:
        """
        Collects demonstrations from the human in each of the specified worlds. Demonstration is saved into the "data"
        directory. The name of each demonstration is returned in a list.

        :param num: number of demonstrations to collect.
        :param w_lst: list of world objects.
        :param fixed_ctrl_lst: list of fixed controls for other car in each world.
        :return:
        """
        names = []
        for i in range(num):
            w = world_lst[i]
            fixed_ctrl = fixed_ctrl_lst[i]
            w.cars[0] = car.UserControlledCar(self.dynamics_function, w.cars[0].x, color="orange")
            w.cars[1] = car.UserControlledCar(self.dynamics_function, w.cars[1].x, color="white")
            w.cars[1].fix_control(fixed_ctrl)

            vis = visualize.Visualizer(self.dt, magnify=1.2, iters=self.time_steps)
            vis.autoquit = True
            vis.use_world(w)
            vis.main_car = w.cars[0]
            vis.run()
            name = "demo-%8f"%(time.time())
            vis.save_trajectory(name)
            names.append(name)
        return names

    def tt_features(self, x: np.ndarray, other_xs: np.ndarray):
        # staying in lane (higher is better)
        def stay_in_lane(x):
            return tt.exp(-30. * tt.min([
                (x[0] - 0.17) ** 2,
                x[0] ** 2,
                (x[0] + 0.17) ** 2,
            ]))

        def keep_speed(x):
            return -0.25 * (x[3] + 1) ** 2

        # heading (higher is better)
        def keep_heading(x):
            return tt.sin(x[2])

        # other_xs[other_index][state] there is no time component.
        def avoid_collisions_with(other_xs, num_others):
            def avoid_collision(x):
                return tt.sum(
                    [tt.exp(-(14 * (x[0] - other_xs[i][0]) ** 2 +
                              6 * (x[1] - other_xs[i][1]) ** 2)) - 1
                     for i in range(num_others)
                     ]
                )
            return avoid_collision

        return tt.stack([
            stay_in_lane(x),
            keep_speed(x),
            keep_heading(x),
            avoid_collisions_with(other_xs, self.num_others)(x),
        ])


    def np_features(self, t: traj.Trajectory) -> np.ndarray:
        def np_stay_in_lane(x):
            return np.exp(-30. * np.min([
                (x[0] - 0.17) ** 2,
                x[0] ** 2,
                (x[0] + 0.17) ** 2,
            ]))

        # if w is negative, best is to set x[3] = 1
        def np_keep_speed(x):
            return -0.25 * (x[3] + 1) ** 2

        def np_keep_heading(x):
            return np.sin(x[2])

        # if w is negative, best is to set x far from other xs
        def np_avoid_collisions_with(other_xs, num_others):
            def avoid_collision(x):
                return np.sum(
                    [np.exp(-(14 * (x[0] - other_xs[i][0]) ** 2 +
                              6 * (x[1] - other_xs[i][1]) ** 2)) - 1
                     for i in range(num_others)
                     ]
                )
            return avoid_collision

        xs = t.states[0]
        other_xs_lst = t.states[1:]
        lst_of_features = []
        for i in range(len(xs)):
            phi = np.stack([
                np_stay_in_lane(xs[i]),
                np_keep_speed(xs[i]),
                np_keep_heading(xs[i]),
                np_avoid_collisions_with(other_xs_lst[:,i,:], len(other_xs_lst[:,i,:]))(xs[i]),
            ])
            lst_of_features.append(phi)
        phi = np.sum(lst_of_features, axis=0)
        return phi
