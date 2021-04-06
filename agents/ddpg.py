# Adopted from https://keras.io/examples/rl/ddpg_pendulum/

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from agents.nets.actor_network import generate_actor_network
from agents.nets.critic_network import generate_critic_network
from agents.util import * 
import tqdm
import time



class DDPGAgent():
    def  __init__(
           self,
           env,
           gamma=0.99,
           tau=0.005,
           critic_lr=0.002,
           actor_lr=0.001,
           noise_std_dev=0.02,
           buffer_size=50000,
           batch_size=64,
           ):

        self._env = env
        self._num_obs = env.observation_space.shape[0]
        self._num_act = env.action_space.shape[0]
        self._max = env.action_space.high[0]
        self._min = env.action_space.low[0]

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

        self._actor_best = generate_actor_network(self._num_obs, self._num_act, self._max)
        self._critic_best = generate_critic_network(self._num_obs, self._num_act)

        self._actor_best_average = generate_actor_network(self._num_obs, self._num_act, self._max)
        self._critic_best_average = generate_critic_network(self._num_obs, self._num_act)

        self._actor_model.summary()

        # Making the weights equal initially
        self._target_actor.set_weights(self._actor_model.get_weights())
        self._target_critic.set_weights(self._critic_model.get_weights())

    def policy(self, state, noise_object, policy_param = 'last'):
        if(policy_param == 'last'):
            sampled_actions = tf.squeeze(self._actor_model(state))
        elif(policy_param == 'best_overall'):
            sampled_actions = tf.squeeze(self._actor_best(state))
        elif(policy_param == 'best_average'):
            sampled_actions = tf.squeeze(self._actor_best_average(state))
        else:
            sampled_actions = tf.squeeze(self._actor_model(state))
            
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

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

    def train(self, num_episodes=100, num_steps=100):
        ep_reward_list =[]
        avg_reward_list = []
        info_buffer = []
        best_reward = None
        best_reward_avg = None

        with tqdm.trange(num_episodes) as t:
            for i in t:
                episodic_reward, info = self.train_episode(num_steps)
                # Save best model overall
                if(best_reward == None or episodic_reward > best_reward):
                    self._actor_best.set_weights(self._actor_model.get_weights())
                    self._critic_best.set_weights(self._critic_model.get_weights())
                    best_reward = episodic_reward
                    #print('\r\nUpdating best model! best_reward = {}\r\n'.format(best_reward))

                ep_reward_list.append(episodic_reward)
                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-40:])
                #print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))

                # Save best model on average
                if(i >= 40 and (best_reward_avg == None or avg_reward > best_reward_avg)):
                    self._actor_best_average.set_weights(self._actor_model.get_weights())
                    self._critic_best_average.set_weights(self._critic_model.get_weights())
                    best_reward_avg = avg_reward
                    #print('\r\nUpdating best model average! best_reward_avg = {}\r\n'.format(best_reward_avg))

                t.set_description(f'Episode {i}')
                episodic_reward_str = "⠀{:6.0f}".format(episodic_reward) #Extra spacing was ignored. So I added an invisible character right before
                avg_reward_str = "⠀{:6.0f}".format(avg_reward)
                t.set_postfix(episodic_reward=episodic_reward_str, average_reward=avg_reward_str)
                avg_reward_list.append(avg_reward)
                info_buffer.append(info)
           
        return ep_reward_list, avg_reward_list, info_buffer

    def train_episode(self, num_steps=100, render=False):
        prev_state = self._env.reset()
        prev_state = prev_state
        episodic_reward = 0
        episode_info = []

        for i in range(num_steps):
            if render:
                self._env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = self.policy(tf_prev_state, self._noise)
            # Recieve state and reward from environment.
            state, reward, done, info = self._env.step(action)
            action = action[0]
            reward = tf.cast(reward, dtype=tf.float32)
            #print(state) 
            self._buffer.add((prev_state, action, reward, state, done))
            episodic_reward += reward

            self.update(*self._buffer.sample_batch(self._batch_size))

            self.update_target(self._target_actor.variables, self._actor_model.variables)
            self.update_target(self._target_critic.variables, self._critic_model.variables)

            # End this episode when `done` is True
            #if done:
                #break
            prev_state = state
            episode_info.append(info)

        return episodic_reward, episode_info
    
    def run_episode(self, num_steps=100, render=True, waitTime=0, policy_param='last', mode='human'):
        prev_state = self._env.reset()
        prev_state = prev_state
        episodic_reward = 0
        episode_info = []
        states = []
        images = []
        for i in range(num_steps):
            #print("Step {}".format(i))
            if render:
                img = self._env.render(mode)
                images.append(img[0])
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            def zero_noise(): # Workaround until we have a better way to call the policy without noise
                return 0
            action = self.policy(tf_prev_state, zero_noise, policy_param) # TODO: Change how we pass noise
            # Recieve state and reward from environment.
            state, reward, done, info = self._env.step(action)
            reward = tf.cast(reward, dtype=tf.float32)
            #print(state) 
            episodic_reward += reward
            # End this episode when `done` is True
            #if done:
                #break
            prev_state = state
            time.sleep(waitTime)
            episode_info.append(info)
            states.append(state)
        return states, episodic_reward, episode_info, images

    def save_models(self, suffix=""):
        self._actor_model.save_weights("./weights/ddpg" + suffix + "/actor")
        self._critic_model.save_weights("./weights/ddpg" + suffix + "/critic")
        self._target_actor.save_weights("./weights/ddpg" + suffix + "/target_actor")
        self._target_critic.save_weights("./weights/ddpg" + suffix + "/target_critic")

    def load_models(self, suffix=""):
        self._actor_model.load_weights("./weights/ddpg" + suffix + "/actor")
        self._critic_model.load_weights("./weights/ddpg" + suffix + "/critic")
        self._target_actor.load_weights("./weights/ddpg" + suffix + "/target_actor")
        self._target_critic.load_weights("./weights/ddpg" + suffix + "/target_critic")

    
