import numpy as np
import tensorflow as tf
import tqdm
from agents.dec_ddpg import DecDDPGAgent
import time
# A class to run an environment and sample actions, distribute rewards, observations, etc.
# Assumes fully decentralized agents ie: no MADDPG(because of a centralized critic)
class DecDDPGRunner():
    def __init__(self,
            env,
            ):
        self.env = env
        # Number of unique observations=number of agents
        self.num_agents = len(env.observation_space)
        self.agents = []
        for i in range(self.num_agents):
            # TODO: Add an interface to modify individual hyperparameters
            agent = DecDDPGAgent(env.observation_space[i], env.action_space[i])
            self.agents.append(agent)

    def get_acts(self, states):
        acts = []
        for i in range(self.num_agents):
            state = tf.expand_dims(tf.convert_to_tensor(states[i]), 0)
            state = tf.reshape(state, (1, -1))
            acts.extend(self.agents[i].policy(state))

        return tf.convert_to_tensor(acts)

    def get_non_exploring_acts(self, states):
        acts = []
        for i in range(self.num_agents):
            state = tf.expand_dims(tf.convert_to_tensor(states[i]), 0)
            state = tf.reshape(state, (1, -1))
            acts.extend(self.agents[i].non_exploring_policy(state))

        return tf.convert_to_tensor(acts)


    def distribute_to_buffers(self, prev_states, actions, rewards, states, dones):
        for i in range(self.num_agents):
            prev_state = np.squeeze(np.reshape(prev_states[i], (-1, 1)))
            state = np.squeeze(np.reshape(states[i], (-1, 1)))
            self.agents[i].push_to_buffer(prev_state, actions[i], rewards[i], state, dones[i])

    def update_agents(self):
        for i in range(self.num_agents):
            self.agents[i].perform_update_step()

    def is_done(self, dones):
        for i in range(len(dones)):
            if dones[i]:
                return True
        return False

    def train(self, num_episodes=100, num_steps=100):
        ep_reward_list =[]
        avg_reward_list = []
        info_buffer = []
 
        with tqdm.trange(num_episodes) as t:
            for i in t:
                episodic_reward, info = self.train_episode(num_steps)
                ep_reward_list.append(episodic_reward)
                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-40:])
                t.set_description(f'Episode {i}')
                episodic_reward_str = "⠀{:6.0f}".format(episodic_reward) #Extra spacing was ignored. So I added an invisible character right before
                avg_reward_str = "⠀{:6.0f}".format(avg_reward)
                t.set_postfix(episodic_reward=episodic_reward_str, average_reward=avg_reward_str)
                avg_reward_list.append(avg_reward)
                info_buffer.append(info)
           
        return ep_reward_list, avg_reward_list, info_buffer

    def train_episode(self, num_steps, render=False):
        prev_states = self.env.reset()
        episodic_reward = 0
        episode_info = []

        for i in range(num_steps):
            if render:
                self.env.render()

            actions = self.get_acts(prev_states)

            # Recieve state and reward from environment.
            states, rewards, dones, info = self.env.step(actions)
            actions = np.squeeze(np.reshape(actions, (-1,1)))
            rewards = tf.cast(rewards, dtype=tf.float32)
            self.distribute_to_buffers(prev_states, actions, rewards, states, dones)
            self.update_agents()
            episodic_reward += sum(rewards)
            episode_info.append(info)

            prev_states = states

            # End this episode when `done` is True
            if self.is_done(dones):
                break

        return episodic_reward, episode_info

    def run_episode(self, num_steps=100, render=True, waitTime=0.1, policy_param='last', mode='human'):
        prev_states = self.env.reset()
        episodic_reward = 0
        episode_info = []
        states = []
        images = []

        for i in range(num_steps):
            if render:
                img = self.env.render(mode)
                images.append(img[0])
            actions = self.get_non_exploring_acts(prev_states)
            states.append(states)
            # Recieve state and reward from environment.
            state, rewards, dones, info = self.env.step(actions)
            rewards = tf.cast(rewards, dtype=tf.float32)
            episodic_reward += sum(rewards)
            episode_info.append(info)

            # End this episode when `done` is True
            if self.is_done(dones):
                break
            prev_states = state
            time.sleep(waitTime)
            episode_info.append(info)

        return states, episodic_reward, episode_info, images

    def save_agents(self, suffix=""):
        for i, agent in enumerate(self.agents):
            agent.save_models(suffix=(suffix + str(i)))

    def save_agents(self, suffix=""):
        for i, agent in enumerate(self.agents):
            agent.load_models(suffix=(suffix + str(i)))
