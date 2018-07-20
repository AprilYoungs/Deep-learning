import gym
import sys
import numpy as np
from agents.agent import DDPG

# init anv and agent 
env = gym.make('Pendulum-v0')
agent = DDPG(env, True)

# get trained weight
path = sys.argv[-1]
agent.load_weight(path)

# visualize the interaction of the agent
state = agent.reset_episode()
iteration = 0
while True:
    env.render()
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    iteration += 1
    if done:
        env.close()
        print('iteraiton-->', iteration)
        break