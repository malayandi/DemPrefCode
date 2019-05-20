import numpy as np


class Trajectory(object):
    """
        Use Trajectory as a container for a trajectory of states and 
        corresponding controls for multiple agents in a system.

        The agent of interest, often the human demonstrating, or the agent
        for which an algorithm is generating behavior, is by convention
        the first agent in the list (i.e., index 0).
    """

    def __init__(self, states: np.ndarray, controls: np.ndarray, null: bool = False) -> None:
        """
            states should be a three dimensional array [agent][time][state]
            controls should be a three dimensional array [agent][time][control]
        """
        self.null = null
        if not self.null:
            if len(states.shape) != 3:
                raise Exception("Trajectory.__init__: states should \
                                    have shape [agent][time][state]")
            if len(controls.shape) != 3:
                raise Exception("Trajectory.__init__: controls should \
                                    have shape [agent][time][control]")
            if states.shape[1] != controls.shape[1]:
                raise Exception("Trajectory.__init__: states and controls should \
                                    have same number of time steps")
            if states.shape[0] != controls.shape[0]:
                raise Exception("Trajectory.__init__: states and controls should \
                                    have same number of agents")
            self.states = states
            self.controls = controls

    def length(self) -> int:
        """
            Use length to measure length of trajectory. Don't use len().
            A trajectory may have zero length.

            >>> t = Trajectory(states, controls)
            >>> t.length()
            10
        """
        return self.states.shape[1]

    def num_agents(self) -> int:
        """
            Use num_agents to check number of agents in trajectory.

            >>> T = Trajectory(states, controls)
            >>> t.num_agents()
            3
        """
        return self.states.shape[0]

    def trim(self, length: int, start: int = 0) -> 'Trajectory':
        """
            Use trim a get sub-trajectory.
            
            >>> states = np.array([[[1], [2]], [[3], [4]]]) # Toy example
            >>> controls = states
            >>> t = Trajectory(states, controls)
            >>> t.length()
            2
            >>> short_t = t.trim(1, start=1)
            >>> short_t.length()
            1
            >>> Trajectory(None, None).length()
            0
        """
        if length < 0:
            raise IndexError("Traj.trim: length must be positive")
        if start < 0:
            raise IndexError("Traj.trim: start must be a positive integer")
        if start + length > self.length():
            raise IndexError("Traj.trim: (start + length) too long")
        return Trajectory(self.states[:,start:start+length,:],
                          self.controls[:,start:start+length,:])
