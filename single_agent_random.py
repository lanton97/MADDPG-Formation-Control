import time
import numpy as np
from agents import ddpg
import matplotlib.pyplot as plt
from gym import spaces
from envs.centralized_env import CentralizedEnvWrapper
from envs.cont_environment import ContMultiAgentEnv
import multiagent.scenarios as scenarios
import tensorflow as tf
import envs.scenarios as scenarios
import imageio

# Things to talk 
# single_agent_test modifications
# results
# same equilibriu point? (good thing)
# getting the best single reward?
# why is it not much better? 
# Testi with fixed particle and landmark location and then increase complexity
# Test with linear neural network


# Adapted from the multi-agent particle env to use a continuous environment
def save_render(path,images):
    with imageio.get_writer(path, mode='I') as writer:
        for img in images:
            writer.append_data(img)

def make_env(scenario_name, benchmark=False):

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    scenario.maxVelAndSensitivity(world, 1,3)
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
images = []
for i in range(0,200):
    img = env.render(mode='rgb_array')
    state, reward, done, info = env.step([env.action_space.sample()]) # take a random action
    rewards.append(reward)
    images.append(img[0])
env.close()
save_render('test.gif', images)

