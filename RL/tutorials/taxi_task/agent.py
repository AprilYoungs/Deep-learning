import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6, epsilon=None, alpha=0.2):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = 1

        self.fix_epsilon = (epsilon != None)
        print('Alpha:{}'.format(self.alpha))

    def get_prob(self, state, i_episode=None):
        """ Given the state, return the probilities of actions
        """

        if i_episode != None and not self.fix_epsilon:
            self.epsilon = max(0.2-0.001*i_episode, 0)

        probs = np.ones(self.nA)*self.epsilon/self.nA
        probs[np.argmax(self.Q[state])] += 1-self.epsilon
        if (self.Q[state] == 0.0).all():
            probs = np.ones(self.nA)/self.nA
        return probs

    def select_action(self, state, i_episode=None, train=True):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        probs = self.get_prob(state, i_episode)

        if train==True:
            return np.random.choice(self.nA, p=probs)
        else:
            return np.argmax(self.Q[state])

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        expected_Q = np.dot(self.get_prob(state),self.Q[next_state])
#         max_Q = np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha*(reward + self.gamma*expected_Q - self.Q[state][action])
