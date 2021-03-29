# Adopted from https://keras.io/examples/rl/ddpg_pendulum/

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from agents.nets.actor_network import generate_actor_network
from agents.nets.critic_network import generate_critic_network
from agents.util import * 
import tqdm

class DecDDPGAgent():
    def  __init__(
           self,
           obs_space,
           action_space,
           gamma=0.99,
           tau=0.005,
           critic_lr=0.002,
           actor_lr=0.001,
           noise_std_dev=0.02,
           buffer_size=50000,
           batch_size=64,
           ):

        self._num_obs = obs_space.shape[0]
        self._num_act = action_space.shape[0]
        self._max = action_space.high[0]
        self._min = action_space.low[0]

        self._gamma = gamma
        self._tau = tau
        self._actor_opt = tf.keras.optimizers.Adam(actor_lr)
        self._critic_opt = tf.keras.optimizers.Adam(critic_lr)

        self._noise = OUNoise(mean=np.zeros(1), std_dev=float(noise_std_dev) * np.ones(1))
        self._buffer= ReplayBuffer(self._num_obs, self._num_act, buffer_capacity=buffer_size)
        self._batch_size=batch_size

        self._actor_model = generate_actor_network(self._num_obs, self._num_act, self._max)
        self._critic_model = generate_critic_network(self._num_obs, self._num_act)

        self._target_actor = generate_actor_network(self._num_obs, self._num_act, self._max)
        self._target_critic = generate_critic_network(self._num_obs, self._num_act)

        self._actor_model.summary()

        # Making the weights equal initially
        self._target_actor.set_weights(self._actor_model.get_weights())
        self._target_critic.set_weights(self._critic_model.get_weights())

    def policy(self, state):
        sampled_actions = tf.squeeze(self._actor_model(state))
        noise = self._noise()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self._min, self._max)

        return np.squeeze(legal_action)

    def non_exploring_policy(self, state):
        sampled_actions = tf.squeeze(self._actor_model(state))

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self._min, self._max)

        return [np.squeeze(legal_action)]

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):

        #tf.print(type(state_batch), type(reward_batch), type(action_batch))
        #tf.print(type(state_batch[0]), type(reward_batch[0]), type(action_batch[0]))
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self._target_actor(next_state_batch, training=True)
            y = reward_batch + self._gamma * self._target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self._critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self._critic_model.trainable_variables)
        self._critic_opt.apply_gradients(
            zip(critic_grad, self._critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self._actor_model(state_batch, training=True)
            critic_value = self._critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self._actor_model.trainable_variables)
        self._actor_opt.apply_gradients(
            zip(actor_grad, self._actor_model.trainable_variables)
        )

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self._tau + a * (1 - self._tau))

    def push_to_buffer(self, state, action, reward, next_state, done):
        self._buffer.add((state, action, reward, next_state, done))

    def perform_update_step(self):
        self.update(*self._buffer.sample_batch(self._batch_size))

        self.update_target(self._target_actor.variables, self._actor_model.variables)
        self.update_target(self._target_critic.variables, self._critic_model.variables)

