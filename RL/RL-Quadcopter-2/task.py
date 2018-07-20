import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * (len(self.sim.pose) + len(self.sim.v) + len(self.sim.angular_v))
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.runtime = runtime

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def compute_distance(self, pos1, pos2):
        return np.sqrt(sum([(a-b)**2 for a,b in zip(pos1, pos2)]))

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # distance = self.compute_distance(self.sim.pose[:3], self.target_pos)
        # self.z_bonus = self.sim.pose[2] - self.target_pos[2]
        # z_diff = self.target_pos[2] - self.sim.pose[2]
        # z_factor = self.z_bonus if z_diff <= 0 else 1.0
        # reward = 1/(1+distance)

        # reward = 1. - 0.3*(abs(self.sim.pose[:3]-self.target_pos)).sum()

        reward = -min(abs(self.sim.pose[2]-self.target_pos[2]), 20.0)
        if self.sim.pose[2] >= self.target_pos[2]:
            reward += 10

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()

            if self.sim.pose[2] >= self.target_pos[2]:
                done = True
            elif self.sim.time > self.runtime:
                reward -= 10.0
                done = True
            states = np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v))
            pose_all.append(states)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v] * self.action_repeat)
        return state
