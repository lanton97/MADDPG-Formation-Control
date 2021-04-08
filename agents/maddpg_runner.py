import numpy as np
import tensorflow as tf
import tqdm
from agents.maddpg import MADDPGAgent
from agents.util import ReplayBuffer
import time

class MADDPGRunner():
    def __init__(self,
            env,
            batch_size=64
            ):
        self.env = env
        self.batch_size = batch_size
        # Number of unique observations=number of agents
        self.num_agents = len(env.observation_space)
        self.agents = []
        self.obs_space_size = 0
        self.act_space_size = 0
        for i in range(self.num_agents):
            # TODO: Add an interface to modify individual hyperparameters
            agent = MADDPGAgent(env, i)
            self.agents.append(agent)
            self.obs_space_size += env.observation_space[i].shape[0]
            self.act_space_size += env.action_space[i].shape[0]

        self.buffer = ReplayBuffer(self.obs_space_size, self.act_space_size, num_agents=self.num_agents) 

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
            state = states[i]
            acts.extend(self.agents[i].non_exploring_policy(state))

        return tf.convert_to_tensor(acts)

    def distribute_to_buffers(self, prev_states, actions, rewards, states, dones):
        prev_state = np.squeeze(np.reshape(prev_states, (-1, 1)))
        state = np.squeeze(np.reshape(states, (-1, 1)))
        self.buffer.add((prev_state, actions, rewards, state, dones))

    def split_obs(self, observations):
        pointer = 0
        pointer_end = 0
        split_obs = []
        for i in range(self.num_agents):
            pointer_end = pointer + self.env.observation_space[i].shape[0]
            split_obs.append(observations[:, pointer:pointer_end])
            pointer = pointer_end
 
        return split_obs

    def update_agents(self, update_batch, next_acts):
        for i in range(self.num_agents):
            self.agents[i].perform_update_step(update_batch, next_acts)

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
            rewards = tf.cast(rewards, dtype=tf.float32)
            actions = np.squeeze(np.reshape(actions, (-1,1)))
            self.distribute_to_buffers(prev_states, actions, rewards, states, dones)
            update_batch = self.buffer.sample_batch(self.batch_size)
            next_obs = self.split_obs(update_batch[-2])
            next_actions = self.get_non_exploring_acts(next_obs)
            next_actions = np.squeeze(np.reshape(next_actions, (self.batch_size, -1, 1)))
            self.update_agents(update_batch, next_actions)
            episodic_reward += sum(rewards)
            episode_info.append(info)

            prev_states = states

            # End this episode when `done` is True
            if self.is_done(dones):
                break

        #print(prev_states[-1][-2:])

        return episodic_reward, episode_info

    def run_episode(self, num_steps=100, render=True, waitTime=0.1, policy_param='last', mode='human'):
        prev_states = self.env.reset()
        episodic_reward = 0
        episode_info = []
        states = []
        images = []

        for i in range(num_steps):
            print("Step {}".format(i))
            if render:
                img = self.env.render()
                images.append(img[0])
            prev_states = self.split_obs(tf.reshape(prev_states, (1, -1)))
            actions = self.get_non_exploring_acts(prev_states)

            states, rewards, dones, info = self.env.step(actions)
            rewards = tf.cast(rewards, dtype=tf.float32)
            episodic_reward += sum(rewards)
            episode_info.append(info)

            # End this episode when `done` is True
            if self.is_done(dones):
                break
            prev_states = states
            time.sleep(waitTime)
            episode_info.append(info)

        return states, episodic_reward, episode_info, images

    def save_agents(self, suffix=""):
        for i, agent in enumerate(self.agents):
            agent.save_models(suffix=(suffix + str(i)))

    def load_agents(self, suffix=""):
        for i, agent in enumerate(self.agents):
            agent.load_models(suffix=(suffix + str(i)))
