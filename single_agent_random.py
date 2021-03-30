import time
import numpy as np
from agents import ddpg
import matplotlib.pyplot as plt
from gym import spaces
from envs.centralized_env import CentralizedEnvWrapper
from envs.cont_environment import ContMultiAgentEnv
import multiagent.scenarios as scenarios
import tensorflow as tf
import envs as scenarios

# Adapted from the multi-agent particle env to use a continuous environment
def make_env(scenario_name, benchmark=False):

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    scenario.maxVelAndSensitivity(world, 0.1,0.2)
    # create multiagent environment
    if benchmark:        
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, shared_viewer=True)
    return env


# Test Code here
# This uses the centralized wrapper
env = CentralizedEnvWrapper(make_env("simple_custom_vel"))

rewards = []
env.reset()
done = False
while not done:
    env.render()
    state, reward, done, info = env.step([env.action_space.sample()]) # take a random action
    rewards.append(reward)
env.close()

