import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from agents.nets.a2c_net import generate_a2c_net
import math

class A2C_agent:

    def __init__(
            self,
            env,
            discount_rate=0.99,
            learning_rate=0.001,
            ):

        self.env = env

        self.num_acts = env.action_space.shape[0]
        self.obs_dims = env.observation_space.shape[0]

        self.max = env.action_space.high[0]
        self.min = env.action_space.low[0]

        self.gamma = discount_rate

        self.optimizer = RMSprop(learning_rate=learning_rate)
        self.net = generate_a2c_net(self.obs_dims, self.num_acts)


    @tf.function
    def update(self, state, action, reward, next_state, done):

        with tf.GradientTape() as tape:
            # First the critic loss
            _, _, val    = self.net(tf.convert_to_tensor(state), training=True)
            _, _, next_val = self.net(tf.convert_to_tensor(next_state), training=True)
            returns = []
            discounted_sum = 0
            for r in reward[::-1]:
                discounted_sum = r + self.gamma * discounted_sum
                returns.insert(0, discounted_sum)
            #returns = self.gamma * (1-tf.convert_to_tensor(done, dtype=tf.float32)) * next_val + tf.convert_to_tensor(reward, dtype=tf.float32)
            adv = tf.stop_gradient(returns - val)
            val_loss    = tf.reduce_mean(tf.math.square(adv))

            # Now the actor loss
            mu, var, _ = self.net(tf.convert_to_tensor(state))
            entropy = -0.5 * (tf.math.log(2.0*np.pi*var) + 1.0)
            pdf_val = 1.0 / tf.math.sqrt(2.0 * np.pi * var) * tf.math.exp(-tf.math.square(action - mu) / (2.0 * var))
            log_prob =  tf.math.log(pdf_val + 1e-5)


            actor_loss = -tf.math.reduce_mean(log_prob * adv + 1e-4 * entropy)
            loss = actor_loss + 0.5*val_loss

        grads = tape.gradient([actor_loss, val_loss], self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

    # Sample an action from a distribution given means and variances
    def sample_action(self, state):
        tfstate = tf.convert_to_tensor(state)
        mu, var, _ = self.net(state)
        eps = np.random.randn(self.num_acts)
        acts = mu + eps * np.sqrt(var)
        acts = np.clip(acts, self.min, self.max)

        prob = 1.0 / np.sqrt(2.0 * np.pi * var) * np.exp(-np.square(acts - mu) / (2.0 * var))
        # TODO Clip values
        return [tf.squeeze(acts)]

    def train(self, num_episodes=100):
        episode_rewards = []
        avg_rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_rew = 0
            done = False
            actions = []
            states = []
            next_states = []
            rewards = []
            dones = []

            while not done:
                action = self.sample_action(tf.expand_dims(state,0))
                actions.append(action)
                next_state, reward, done, info = self.env.step(action)
                reward = tf.cast(reward, dtype=tf.float32)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

                states.append(state)
                state = next_state

            self.update(states, actions, rewards, next_states, dones)
            episode_rewards.append(np.sum(rewards))
            avg_rewards.append(np.mean(episode_rewards[-40:]))
            print("Average rewards for episode " + str(episode) + ": " + str(avg_rewards[-1]))

        return episode_rewards
