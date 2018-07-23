import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


def plot_rewards(results, agent_name, task_name, zoom_x_range, zoom_y_range, N=10):
    '''Plot the scores curve and the smooth curve'''
    episode_rewards = results.groupby(['episode'])['reward'].sum()
    smoothed_rewards = episode_rewards.rolling(N).mean()
    eps = list(range(1, len(episode_rewards)+1))

    fig, ax = plt.subplots(1, 2, figsize = (18, 6))
    fig.suptitle('Rewards earned by the {} on the {}'.format(agent_name, task_name), fontsize = 18, y=1.05)

    ax[0].plot(smoothed_rewards, label='Running Average of Reward (n={})'.format(N))
    ax[0].plot(episode_rewards, color='grey', alpha=0.3, label='Total Reward in Episode')
    ax[0].set_title("{}:\nTotal Reward per Episode in {}".format(agent_name, task_name))
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel('Reward')
    ax[0].legend()

    ax[1].plot(smoothed_rewards, label='Running Average of Reward (n={})'.format(N))
    ax[1].plot(episode_rewards, color='grey', alpha=0.3, label='Total Reward in Episode')
    ax[1].set_xlim(*zoom_x_range)
    ax[1].set_ylim(*zoom_y_range)
    ax[1].set_title("{}:\nTotal Reward on per Episode in Final {} \Episode of {}".format(agent_name, zoom_x_range[0],task_name))
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel('Reward')
    ax[1].legend()

    plt.show();


def visualize_result(results, target_pos, agent_name):
    """
    plot all the data at one time
    """

    best_reward_episode = results.groupby(['episode'])['reward'].sum().idxmax()
    best_episode = results.query('episode=={}'.format(best_reward_episode))
    best_episode = best_episode.groupby('time')['x','y','z'].mean()

    last_10_percent_episode = results.query('episode >= {}'.format(results.episode.max() * 0.9))
    average_last_10_percent_xyz = results.groupby('time')['x','y','z'].mean()

    # trajectory of best episode
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(best_episode.x.tolist(), best_episode.y.tolist(), best_episode.z.tolist(), label='trajectory')
    ax.plot(best_episode.x.tolist()[:1], best_episode.y.tolist()[:1], best_episode.z.tolist()[:1], 'yx' ,label='start')
    ax.plot(best_episode.x.tolist()[-1:], best_episode.y.tolist()[-1:], best_episode.z.tolist()[-1:], 'co', label='end')
    ax.plot(*np.reshape(target_pos, [len(target_pos), -1]), 'kx', label='target')
    ax.set_title('trajectory of best episode')
    ax.legend()
    plt.show();

    fig, ax = plt.subplots(3, 2, figsize=(18,18))
    fig.suptitle('positions of {}'.format(agent_name), fontsize = 18, y=1.05)

    # position
    ax[0, 0].plot(average_last_10_percent_xyz)
    ax[0, 0].legend(['x','y','z']);
    ax[0, 0].set_title('Position of the average of last 10% episode')
    _ = plt.ylim()


    ax[0, 1].plot(best_episode)
    ax[0, 1].legend(['x','y','z']);
    ax[0, 1].set_title('Position of each timestamp during episode({}) of highest reward'.format(best_reward_episode))
    _ = plt.ylim()

    indexs = [(1,0),(1,1),(2,0),(2,1)]

    for i in range(len(indexs)):
        df = results.query('episode == {}'.format(results.episode.max()-i)).groupby('time')['x','y','z'].mean()
        ax[indexs[i][0],indexs[i][1]].plot(df)
        ax[indexs[i][0],indexs[i][1]].legend(['x','y','z']);
        ax[indexs[i][0],indexs[i][1]].set_title('Position of each timestamp during lasr four episode'.format(best_reward_episode))
    # # velocity
    # plt.plot(results['time'], results['x_velocity'], label='x_hat')
    # plt.plot(results['time'], results['y_velocity'], label='y_hat')
    # plt.plot(results['time'], results['z_velocity'], label='z_hat')
    # plt.title('Velocity')
    # plt.legend()
    # _ = plt.ylim()
    # plt.show()
    #
    # # Euler angles
    # plt.plot(results['time'], results['phi'], label='phi')
    # plt.plot(results['time'], results['theta'], label='theta')
    # plt.plot(results['time'], results['psi'], label='psi')
    # plt.title('Euler angles')
    # plt.legend()
    # _ = plt.ylim()
    # plt.show()
    #
    # # Velocity of euler angles
    # plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    # plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    # plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    # plt.title('Velocity of euler angles')
    # plt.legend()
    # _ = plt.ylim()
    # plt.show()
    #
    # # Rotor_speeds
    # plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    # plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    # plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    # plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    # plt.title('Rotor_speeds')
    # plt.legend()
    # _ = plt.ylim()
    # plt.show()
