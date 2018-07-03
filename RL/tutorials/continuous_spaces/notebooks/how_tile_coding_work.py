import sys
import gym
import numpy as np
import pandas as pd

def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    """Define a uniformly-spaced grid that can be used for tile-coding a space.
    
    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins or tiles along each corresponding dimension.
    offsets : tuple
        Split points for each dimension should be offset by these values.
    
    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    # TODO: Implement this
    print(low,high, bins)
    assert len(low)==len(high) and len(low)==len(bins)
    
    grids = []
    for i in range(len(low)):
        grid = low[i] + offsets[i] + np.arange(1, bins[i])*(high[i]-low[i])/bins[i]
        grids.append(grid)
    
    return grids

def create_tilings(low, high, tiling_specs):
    """Define multiple tilings using the provided specifications.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    tiling_specs : list of tuples
        A sequence of (bins, offsets) to be passed to create_tiling_grid().

    Returns
    -------
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    """
    # TODO: Implement this
    return [create_tiling_grid(low, high, bins, offsets) \
            for bins, offsets in tiling_specs]

def discretize(sample, grid):
    """Discretize a sample as per given grid.
    
    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    
    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    # TODO: Implement this
    return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))


def tile_encode(sample, tilings, flatten=False):
    """Encode given sample using tile-coding.
    
    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    flatten : bool
        If true, flatten the resulting binary arrays into a single long vector.

    Returns
    -------
    encoded_sample : list or array_like
        A list of binary vectors, one for each tiling, or flattened into one.
    """
    # TODO: Implement this
    discretizations = [discretize(sample, grid) for grid in tilings]
    
    return np.hstack(discretizations) if flatten else discretizations

class QTable:
    """Simple Q-table."""

    def __init__(self, state_size, action_size):
        """Initialize Q-table.
        
        Parameters
        ----------
        state_size : tuple
            Number of discrete values along each dimension of state space.
        action_size : int
            Number of discrete actions in action space.
        """
        self.state_size = state_size
        self.action_size = action_size

        # TODO: Create Q-table, initialize all Q-values to zero
        # Note: If state_size = (9, 9), action_size = 2, q_table.shape should be (9, 9, 2)
        self.q_table = np.zeros(state_size+(action_size,))
        
        print("QTable(): size =", self.q_table.shape)


class TiledQTable:
    """Composite Q-table with an internal tile coding scheme."""
    
    def __init__(self, low, high, tiling_specs, action_size):
        """Create tilings and initialize internal Q-table(s).
        
        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of state space.
        high : array_like
            Upper bounds for each dimension of state space.
        tiling_specs : list of tuples
            A sequence of (bins, offsets) to be passed to create_tilings() along with low, high.
        action_size : int
            Number of discrete actions in action space.
        """
        self.tilings = create_tilings(low, high, tiling_specs)
        self.state_sizes = [tuple(len(splits)+1 for splits in tiling_grid) for tiling_grid in self.tilings]
        self.action_size = action_size
        self.q_tables = np.array([np.zeros(state_size+(self.action_size,)) for state_size in self.state_sizes])
        print("TiledQTable(): no. of internal tables = ", len(self.q_tables))
    
    def get(self, state, action):
        """Get Q-value for given <state, action> pair.
        
        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.
        
        Returns
        -------
        value : float
            Q-value of given <state, action> pair, averaged from all internal Q-tables.
        """
        # TODO: Encode state to get tile indices
        discrete_states = tile_encode(state, self.tilings)
        
        # TODO: Retrieve q-value for each tiling, and return their average
        
        return np.mean([q_table[tuple(state)+(action,)] for q_table, state in zip(self.q_tables, discrete_states)])
    
    def get_state(self, state):
        """
            Get all Q-values for the given state
        """
        return [self.get(state, action) for action in range(self.action_size)]
        
    
    def update(self, state, action, value, alpha=0.1):
        """Soft-update Q-value for given <state, action> pair to value.
        
        Instead of overwriting Q(state, action) with value, perform soft-update:
            Q(state, action) = alpha * value + (1.0 - alpha) * Q(state, action)
            Q(state, action) = alpha * value + (Q(state, action) - alpha* Q(state, action)) 
            Q(state, action) = Q(state, action) + alpha * (value - Q(state, action))
        
        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.
        value : float
            Desired Q-value for <state, action> pair.
        alpha : float
            Update factor to perform soft-update, in [0.0, 1.0] range.
        """
        # TODO: Encode state to get tile indices
        discrete_states = tile_encode(state, self.tilings)
        
        # TODO: Update q-value for each tiling by update factor alpha
        for q_table, state in zip(self.q_tables, discrete_states):
            q_table[tuple(state) + (action,)] += alpha * (value - q_table[tuple(state) + (action,)])
    
    def save_to(self,path):
        """Save the Q_table to a specific path ,*.npy"""
        np.save(path, self.q_tables)
    
    def load(self,path):
        """reload q_tables from pre-saved data, *.npy"""
        self.q_tables = np.load(path)

class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by tile coding it ."""
    
    def __init__(self, env, tiling_specs, alpha=None,  gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.999, min_epsilon=0.01, seed=505):
        """Initialize variables, create tilings for discretization."""
        
        self.env = env
        self.tiling_specs = tiling_specs
        self.seed = np.random.seed(seed)
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = self.initial_epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.action_size = env.action_space.n
        
        self.q_table = TiledQTable(env.observation_space.low, env.observation_space.high, tiling_specs, env.action_space.n)
        
    def reset_episode(self, state):
        """Reset variables for a new episode."""
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)
        
        self.last_state = state
        self.last_action = np.argmax(self.q_table.get_state(state))
        return self.last_action
        
    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training"""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon
    
    def act(self, state, reward=None, done=None, mode='train'):
        """
        Pick next action and update internal Q table (when mode != 'test').
        """
        if mode == 'test':
            action = np.argmax(self.q_table.get_state(state))
        else:
            alpha = 0.1 if self.alpha == None else self.alpha
            value = reward + self.gamma*np.max(self.q_table.get_state(state))
            self.q_table.update(self.last_state, self.last_action, value, alpha=alpha)
            
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                action = np.random.randint(0, self.action_size)
            else:
                action = np.argmax(self.q_table.get_state(state))
        
        self.last_state = state
        self.last_action = action
        
        return action

def run(agent, env, num_episodes=20000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False
        
        while not done:
            state, reward, done, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)
        
        scores.append(total_reward)
        
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
                
                if i_episode % 100 == 0:
                    print('\rEpisode {}/{} | Max Average Score: {}'.format(i_episode, num_episodes, max_avg_score), end="")
                    sys.stdout.flush()
    
    return scores

################   render the environment ###################

path = sys.argv[-1]

env = gym.make('Acrobot-v1')
env.seed(505);

q_tiling_spaces = [
                ((10, 10, 10, 10, 10, 10), (-0.066, -0.066, -0.066, -0.066, -0.838, -1.885)),
                ((10, 10, 10, 10, 10, 10), (0, 0, 0, 0, 0, 0)),
                ((10, 10, 10, 10, 10, 10), (0.066, 0.066, 0.066, 0.066, 0.838, 1.885))
                ]
q_agent = QLearningAgent(env, q_tiling_spaces)
q_agent.q_table.load(path)

state = env.reset()
scores = 0
while True:
    action = q_agent.act(state, mode='test')
    state, reward, done, info = env.step(action)
    scores += reward
    env.render()
    if done:
        print('Final scroe:',scores)
        break