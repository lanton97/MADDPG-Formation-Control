# Adapted partiall from https://keras.io/examples/rl/ddpg_pendulum/

import numpy as np
import random
import tensorflow as tf


class ReplayBuffer(object):

    def __init__(self, num_states, num_actions, num_agents=1, buffer_capacity=100000):
        # number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity

        # its tells us num of times record() was called.
        self.buffer_counter = 0

        # instead of list of tuples as the exp.replay concept go
        # we use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, num_agents))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, num_agents))

    # takes (s,a,r,s') obervation tuple as input
    def add(self, obs_tuple):
        # set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    def sample_batch(self, batch_size=64):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices])

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def clear(self):
        self._buffer.clear()


# Class for Ornstein-Uhlenbeck Process
class OUNoise(object):
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2):
        self._theta   = theta
        self._mean    = mean
        self._std_dev = std_dev
        self._dt      = dt
        self.clear()

    def __call__(self):
        x = ( self._x_prev +
              self._theta * (self._mean - self._x_prev) * self._dt +
              self._std_dev * np.sqrt(self._dt) * np.random.normal(size=self._mean.shape)
            )
        self._x_prev = x
        return x

    def clear(self):
        self._x_prev = np.zeros_like(self._mean)

