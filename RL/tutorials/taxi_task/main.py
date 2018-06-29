from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
agent = Agent(alpha=1, epsilon=0.8)
avg_rewards, best_avg_reward = interact(env, agent, num_episodes=10000)

state = env.reset()
env.render()
rewards = 0
while True:
    action = agent.select_action(state)
    state, reward, done, _ = env.step(action)
    env.render()
    rewards += reward
    print('total rewards:' ,rewards)
    if done:
        break


# alpha = np.linspace(0.1, 0.001, num=5, endpoint=True)
# epsilon = np.linspace(0.0001, 0, num=5, endpoint=True)
# for a in alpha:
#     for e in epsilon:
#         interact(env, Agent(epsilon=e, alpha=a))
