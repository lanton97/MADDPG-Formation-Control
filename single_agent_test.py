import time
import numpy as np
from agents import ddpg
import matplotlib.pyplot as plt
from gym import spaces
from envs.centralized_env import CentralizedEnvWrapper
from envs.cont_environment import ContMultiAgentEnv
import multiagent.scenarios as scenarios
import tensorflow as tf

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
env = CentralizedEnvWrapper(make_env("simple"))
agent = ddpg.DDPGAgent(env)
<<<<<<< Updated upstream
print("Training Model")
rewards, avg_rewards = agent.train(500)

input("Press to run trained model") # Even requesting a key press, for us to be prepared to watch the model
agent.run_episode(waitTime = 0.1) # Should we store the model that obtained the best reward? or always use the last one?
=======
rewards, avg_rewards = agent.train(30)

#env.close()
>>>>>>> Stashed changes

#env.render()
#env.close()

print('Finished!')

plt.plot(rewards)
plt.plot(avg_rewards)
plt.show()

