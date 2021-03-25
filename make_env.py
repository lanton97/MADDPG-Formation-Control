import time
import numpy as np
from agents.ddpg import DDPGAgent
import matplotlib.pyplot as plt
from gym import spaces
from envs.centralized_env import CentralizedEnvWrapper
from envs.cont_environment import ContMultiAgentEnv
import envs.scenarios as scenarios

# Adapted from the multi-agent particle env to use a continuous environment
def make_env(scenario_name, benchmark=False):

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


# Test Code here
# This uses the centralized wrapper
env = CentralizedEnvWrapper(make_env("simple_formation"))
agent = DDPGAgent(env)
rewards, avg_rewards = agent.train(500)

plt.plot(rewards)
plt.plot(avg_rewards)
plt.show()

