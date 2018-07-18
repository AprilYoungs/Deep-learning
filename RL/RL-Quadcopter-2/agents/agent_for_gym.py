import random
from collections import namedtuple, deque
from keras import layers, models, optimizers
from keras import backend as K
import numpy as np
import copy
import pickle
import os


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

Experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # set.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])


    def add(self, state, action, reward, next_state, done):
        """Add new experience to memory"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """Randomly sample a batch of experience from memory"""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model

        Params
        ======
            state_size (int):Dimension of each state
            aciton_size (int):Dimension of each action
            action_low (array):Min values of acitons
            aciton_high (array):Max values of actions
        """
        self.state_size = state_size
        self.aciton_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        ###

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        states = layers.Input(shape=(self.state_size, ), name='states')


        net = layers.Dense(units=64, activation='relu')(states)
        net = layers.Dense(units=128, activation='relu')(net)


        # narrow the network to the action size
        raw_actions = layers.Dense(units=self.aciton_size, activation='tanh', name='raw_actions')(net)

        # Scale outpout to proper range
        actions = layers.Lambda(lambda x: (x/2 * self.action_range) + (self.action_low + self.action_high)/2, name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value gradients
        action_gradients = layers.Input(shape=(self.aciton_size, ))

        #### Why this function ?? (Q value) gradients
        loss = K.mean(-action_gradients * actions)

        # Any other Loss

        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
                        inputs=[self.model.input, action_gradients, K.learning_phase()],
                        outputs=[],
                        updates=updates_op)

class Critic:

    def __init__(self, state_size, action_size):
        """Critic (value) Model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each state
        """

        self.state_size = state_size
        self.action_size = action_size

        # Any other parameters

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Input layers
        states = layers.Input(shape=(self.state_size, ), name='states')
        actions = layers.Input(shape=(self.action_size, ), name='actions')

        net_states = layers.Dense(units=64, activation='relu')(states)
        net_states = layers.Dense(units=128, activation='relu')(net_states)

        net_actions = layers.Dense(units=64, activation='relu')(actions)
        net_actions = layers.Dense(units=128, activation='relu')(net_actions)

        # Try different layers

        #Combine state and action
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients
        action_gradients = K.gradients(Q_values, actions)

        self.get_action_gradients = K.function(
                    inputs=[*self.model.input, K.learning_phase()],
                    outputs=action_gradients)

class DDPG():
    """Reinforcement learning agent that learns using DDPG"""
    def __init__(self, task):
        self.task = task
        self.state_size = task.observation_space.shape[0]
        self.action_size = task.action_space.shape[0]
        self.action_low = task.action_space.low
        self.action_high = task.action_space.high

        # Actor
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model params with local model params
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        #Noise Process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        #Replay memory
        self.buffer_size = int(1e5)
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99 # dicount factor
        self.tau = 0.001 # for soft update of target parameters

        self.best_score = -np.inf


    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.score = 0
        return state

    def step(self, action, reward, next_state, done):

        # save experience
        self.memory.add(self.last_state, action, reward, next_state, done)
        self.score += reward

        if done:
            self.best_score = max(self.best_score, self.score)

        #Learn, if enough samples are available
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Return actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample()) # noise for exploration

    def learn(self, experience):
        """Update policy and value parameters using given batch of experience tuples."""

        # Convert experiences to separate list
        states = np.vstack([e.state for e in experience if e is not None])
        actions = np.array([e.action for e in experience if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experience if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experience if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experience if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        # soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size."

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def save_weight(self, path):
        '''the parent path to save the actor and critic weight,
            the weights will be save respectively'''
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(path+"/action_weight", 'wb') as file:
            pickle.dump(self.actor_local.model.get_weights(), file)
        with open(path+"/critic_weight", 'wb') as file:
            pickle.dump(self.critic_local.model.get_weights(), file)

    def load_weight(self, path):
        '''load the previous save weights, path is the parent path of the weights'''
        with open(path+"/action_weight", 'rb') as fi:
            self.actor_local.model.set_weights(pickle.load(fi))
        with open(path+"/critic_weight", 'rb') as file:
            print(file)
            self.critic_local.model.set_weights(pickle.load(file))

        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
