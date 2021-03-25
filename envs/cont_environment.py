# Modified from https://github.com/openai/multiagent-particle-envs
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.environment import MultiAgentEnv

# TODO: Clean this to use just the super init and change the one parameter we need

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class ContMultiAgentEnv(MultiAgentEnv):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        super().__init__(world, reset_callback, reward_callback, observation_callback, 
                info_callback, done_callback, shared_viewer)

        self.discrete_action_input = False
        self.force_discrete_action = False        # configure spaces
        self.action_space = []
        self.observation_space = []

        for agent in self.agents:
            total_action_space = []
            u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        self._reset_render()


    # We override the 
    def step(self, action_n):
        obs_n, reward_n, done_n, info_n = super().step(action_n)
        for agent in self.agents:
            info_n.append(agents.state.p_pos)

        return obs_n, reward_n, done_n, info_n
