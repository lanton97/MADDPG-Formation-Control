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

# Adapted from the multi-agent particle env to use a continuous environment
def make_env(scenario_name, benchmark=False):

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    scenario.maxVelAndSensitivity(world, 0.1,1)
    # create multiagent environment
    if benchmark:        
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, shared_viewer=False)
    return env


# Test Code here
# This uses the centralized wrapper
env = CentralizedEnvWrapper(make_env("simple_custom_vel"))
agent = ddpg.DDPGAgent(env)

print("Training Model")
rewards, avg_rewards, info = agent.train(num_episodes=500, num_steps=100)

input("Press to run trained model") # Even requesting a key press, for us to be prepared to watch the model
agent.run_episode(500, waitTime = 0.05) # Should we store the model that obtained the best reward? or always use the last one?
agent.run_episode(500, waitTime = 0.05) # Should we store the model that obtained the best reward? or always use the last one?
agent.run_episode(500, waitTime = 0.05, best=True) # Should we store the model that obtained the best reward? or always use the last one?
agent.run_episode(500, waitTime = 0.05, best=True) # Should we store the model that obtained the best reward? or always use the last one?
print('Finished!')


print(info[-1])

plt.axhline(y=0, color='r', linestyle='--')
plt.plot(rewards)
plt.plot(avg_rewards)
plt.show()

