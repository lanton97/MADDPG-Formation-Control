# Adopted from https://keras.io/examples/rl/ddpg_pendulum/

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from agents.nets.actor_network import generate_actor_network
from agents.nets.critic_network import generate_critic_network
from agents.util import * 
import tqdm

# Our basic MADDPG: Currenty assumes access to all other agents actions/policies and observations while training
# Could be interesting to explore their policy estimation and ensemble suggestions
class MADDPGAgent():
    def  __init__(
           self,
           env,
           agent_index,
           gamma=0.95,
           tau=0.01,
           critic_lr=0.01,
           actor_lr=0.01,
           noise_std_dev=0.02,
           buffer_size=10e6,
           batch_size=1024,
           ):

        self._agent_index = agent_index
        self._num_agents = len(env.observation_space)

        # With MADDPG, these are only used for the actor
        self._num_obs = env.observation_space[agent_index].shape[0]
        self._num_act = env.action_space[agent_index].shape[0]
        self._max = env.action_space[agent_index].high[0]
        self._min = env.action_space[agent_index].low[0]

        self._gamma = gamma
        self._tau = tau
        self._actor_opt = tf.keras.optimizers.Adam(actor_lr)
        self._critic_opt = tf.keras.optimizers.Adam(critic_lr)

        self._noise = OUNoise(mean=np.zeros(1), std_dev=float(noise_std_dev) * np.ones(1))
        self._actor_model = generate_actor_network(self._num_obs, self._num_act, self._max)

        # Iterate through the environment spaces to figure out the total action and obs dimensionality
        # This should work for agents with variable obs/act spaces, though this isn't the case here
        self._total_obs_size = 0
        self._total_act_size = 0
        for i in range(self._num_agents):
            if i == agent_index:
                self._obs_start_ind = self._total_obs_size
                self._act_start_ind = self._total_act_size
            self._total_obs_size += env.observation_space[i].shape[0]
            self._total_act_size += env.action_space[i].shape[0]

        self._critic_model = generate_critic_network(self._total_obs_size, self._total_act_size)

        self._target_actor = generate_actor_network(self._num_obs, self._num_act, self._max)
        self._target_critic = generate_critic_network(self._total_obs_size, self._total_act_size)

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

        return [np.squeeze(legal_action)]

    def non_exploring_policy(self, state):
        sampled_actions = tf.squeeze(self._actor_model(state))

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self._min, self._max)

        return [np.squeeze(legal_action)]

    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch, next_actions
    ):
        local_next_state_batch = next_state_batch[:,self._obs_start_ind:self._obs_start_ind + self._num_obs]
        with tf.GradientTape() as tape:
            target_actions = self._target_actor(local_next_state_batch, training=True)
            updated_next_actions = tf.concat([next_actions[:,:self._act_start_ind], target_actions[:], next_actions[:,self._act_start_ind + self._num_act:]], 1)
            r = tf.expand_dims(reward_batch[:,self._agent_index], -1)
            y = r + self._gamma * self._target_critic(
                [next_state_batch, updated_next_actions], training=True
            )

            critic_value = self._critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self._critic_model.trainable_variables)
        self._critic_opt.apply_gradients(
            zip(critic_grad, self._critic_model.trainable_variables)
        )

        local_state_batch = state_batch[:,self._obs_start_ind:self._obs_start_ind + self._num_obs]

        with tf.GradientTape() as tape:
            local_actions = tf.cast(self._actor_model(local_state_batch, training=True), dtype=tf.float64)
            updated_action = tf.concat([action_batch[:,0:self._act_start_ind], local_actions[:], action_batch[:,self._act_start_ind + self._num_act:]], 1)
            critic_value = self._critic_model([state_batch, updated_action], training=True)
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

    def perform_update_step(self, update_batch, next_actions):
        self.update(*update_batch, next_actions)

        self.update_target(self._target_actor.variables, self._actor_model.variables)
        self.update_target(self._target_critic.variables, self._critic_model.variables)

