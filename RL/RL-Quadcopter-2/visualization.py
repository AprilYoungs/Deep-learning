import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

def plot_rewards(scores, N=10):
    '''Plot the scores curve and the smooth curve'''
    smoothed_scores = running_mean(scores, N)
    eps = list(range(1, len(scores)+1))
    plt.plot(eps[-len(smoothed_scores):], smoothed_scores)
    plt.plot(eps, scores, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('reward')
    

def visualize_result(results, target_pos):
    """
    plot all the data at one time
    """
    # trajectory
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(results['x'], results['y'], results['z'], label='trajectory')
    ax.plot(results['x'][:1], results['y'][:1], results['z'][:1], 'yx' ,label='start')
    ax.plot(results['x'][-1:], results['y'][-1:], results['z'][-1:], 'co', label='end')
    ax.plot(*np.reshape(target_pos, [len(target_pos), -1]), 'kx', label='target')
    ax.legend()
    plt.show();
    
    # position
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.title('Position')
    plt.legend()
    _ = plt.ylim()
    plt.show()
    
    # velocity
    plt.plot(results['time'], results['x_velocity'], label='x_hat')
    plt.plot(results['time'], results['y_velocity'], label='y_hat')
    plt.plot(results['time'], results['z_velocity'], label='z_hat')
    plt.title('Velocity')
    plt.legend()
    _ = plt.ylim()
    plt.show()
    
    # Euler angles
    plt.plot(results['time'], results['phi'], label='phi')
    plt.plot(results['time'], results['theta'], label='theta')
    plt.plot(results['time'], results['psi'], label='psi')
    plt.title('Euler angles')
    plt.legend()
    _ = plt.ylim()
    plt.show()
    
    # Velocity of euler angles
    plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    plt.title('Velocity of euler angles')
    plt.legend()
    _ = plt.ylim()
    plt.show()
    
    # Rotor_speeds
    plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    plt.title('Rotor_speeds')
    plt.legend()
    _ = plt.ylim()
    plt.show()