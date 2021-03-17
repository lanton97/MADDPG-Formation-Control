# Modified from https://github.com/openai/multiagent-particle-envs
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from envs.cont_environment import ContMultiAgentEnv


class CentralizedEnvWrapper():

    def __init__(self, env):
        self.env = env
        # We assume that each observation space is identical
        num_obs  = len(env.observation_space)
        single_obs = env.observation_space[0].shape[0]
        total_obs = single_obs*num_obs
        #shape = (*single_obs_shape, num_obs,)
        lows = np.reshape(np.array([a.low for a in env.observation_space]), (total_obs,))
        highs = np.reshape(np.array([a.high for a in env.observation_space]), (total_obs,))
        self.observation_space = spaces.Box(lows, highs)#, shape=shape)

        self.num_agents = len(env.action_space)
        self.acts_per_agent = env.action_space[0].shape[0]
        total_acts = self.num_agents*self.acts_per_agent
        act_lows = np.reshape(np.array([a.low for a in env.action_space]), (total_acts,))
        act_highs = np.reshape(np.array([a.high for a in env.action_space]), (total_acts,))
        self.action_space = spaces.Box(act_lows, act_highs)#, shape=shape)
    
    def join_lists(self, lists):
        out_list = []
        for list in lists:
            out_list.extend(list)
        return out_list

    def split_actions(self, actions):
        out_actions = []
        for i in range(self.num_agents):
            agent_acts = actions[0][i*self.acts_per_agent:(i + 1)*self.acts_per_agent]
            out_actions.append(agent_acts)
        return out_actions

    def reset(self):
        states = self.env.reset()
        return self.join_lists(states)

    def step(self, action):
        split_acts = self.split_actions(action)
        state, reward, done, info = self.env.step(split_acts)
        state = self.join_lists(state)
        reward = np.sum(reward)
        done   = done[0]

        return state, reward, done, info
        
    def render(self):
        self.env.render()
