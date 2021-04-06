import time
import numpy as np
from agents.ddpg import DDPGAgent
import matplotlib.pyplot as plt
from gym import spaces
from envs.centralized_env import CentralizedEnvWrapper
from envs.cont_environment import ContMultiAgentEnv
import envs.scenarios as scenarios
from agents.dec_ddpg_runner import DecDDPGRunner
from agents.maddpg_runner import MADDPGRunner

# Adapted from the multi-agent particle env to use a continuous environment
def make_env(scenario_name, benchmark=False):

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.done)
    else:
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done)
    return env


# Test Code here
env = make_env("simple_formation")
dec_agents = MADDPGRunner(env)

rewards, avg_rewards, info = dec_agents.train(500)

dec_agents.save_agents(suffix='formation')

input("press to run trained model") # even requesting a key press, for us to be prepared to watch the model
dec_agents.run_episode()#waittime = 0.1) # should we store the model that obtained the best reward? or always use the last one?

plt.plot(rewards)
plt.plot(avg_rewards)
plt.show()
