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
import os, datetime, imageio
# Adapted from the multi-agent particle env to use a continuous environment
def make_env(scenario_name, benchmark=False):

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    scenario.maxVelAndSensitivity(world, 1,0.5)
    # create multiagent environment
    if benchmark:        
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, shared_viewer=False)
    return env

def generate_path(dir):
    exp_dir = os.path.join(dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))
    os.makedirs(exp_dir)
    return exp_dir

def plot_train_data(rewards, avg_rewards, path =[]):
    fig, ax = plt.subplots()  #create figure and axes
    plt.axhline(y=0, color='r', linestyle='--', label='Zero line')
    plt.plot(rewards, label='Rewards')
    plt.plot(avg_rewards, label='Average rewards')

    handles, _ = ax.get_legend_handles_labels()
    # Slice list to remove first handle
    plt.legend(handles = handles[:])
    plt.title('Train History')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.savefig(path)

def plot_episode_data(states, path = []):
    x = [item[2] for item in states]
    y = [item[3] for item in states]
    fig, ax = plt.subplots()  #create figure and axes
    plt.plot(x, y,'--')
    plt.plot(x[-1], y[-1], 'or')
    plt.xlim((-3,3))
    plt.ylim((-3,3))
    plt.title("Particle Trajectory")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(path)

def save_render(path,images):
    print("Saving Gif!!!")
    with imageio.get_writer(path, mode='I') as writer:
        for img in images:
            writer.append_data(img)
    print("Finished saving Gif")

# Start Experiment
# Experiment path
dir = generate_path("./experiments/single_particle/")


env = CentralizedEnvWrapper(make_env("simple_custom_vel"))
agent = ddpg.DDPGAgent(env)


num_episodes = 1000
num_steps = 300

print("Training Model")
rewards, avg_rewards, info = agent.train(num_episodes=num_episodes, num_steps=num_steps)

input("Press to run trained model") # Even requesting a key press, for us to be prepared to watch the model
states_last_1, episodic_reward_last_1, info_last_1, images_last_1 = agent.run_episode(num_steps, waitTime = 0,mode='rgb_array') # Should we store the model that obtained the best reward? or always use the last one?
states_last_2, episodic_reward_last_2, info_last_2, images_last_2 = agent.run_episode(num_steps, waitTime = 0,mode='rgb_array')  # Should we store the model that obtained the best reward? or always use the last one?
states_overall_1, episodic_reward_overall_1, info_last_overall_1, images_overall_1 = agent.run_episode(num_steps, waitTime = 0, mode='rgb_array', policy_param='best_overall') # Should we store the model that obtained the best reward? or always use the last one?
states_overall_2, episodic_reward_overall_2, info_last_overall_2, images_overall_2 = agent.run_episode(num_steps, waitTime = 0, mode='rgb_array', policy_param='best_overall') # Should we store the model that obtained the best reward? or always use the last one?
states_average_1, episodic_reward_average_1, info_last_average_1, images_average_1 = agent.run_episode(num_steps, waitTime = 0, mode='rgb_array', policy_param='best_average') # Should we store the model that obtained the best reward? or always use the last one?
states_average_2, episodic_reward_average_2, info_last_average_2, images_average_2 = agent.run_episode(num_steps, waitTime = 0, mode='rgb_array', policy_param='best_average') # Should we store the model that obtained the best reward? or always use the last one?
env.close()

save_render(dir + "/images_last_1.gif", images_last_1)
save_render(dir + "/images_last_2.gif", images_last_2)
save_render(dir + "/images_overall_1.gif", images_overall_1)
save_render(dir + "/images_overall_2.gif", images_overall_2)
save_render(dir + "/images_average_1.gif", images_average_1)
save_render(dir + "/images_average_2.gif", images_average_2)

plot_train_data(rewards, avg_rewards, path=dir+'train_data.png')

# TODO: write the episodic reward somewhere?
plot_episode_data(states_last_1, path=dir+'states_last_1.png')
plot_episode_data(states_last_2, path=dir+'states_last_2.png')
plot_episode_data(states_overall_1, path=dir+'states_overall_1.png')
plot_episode_data(states_overall_2, path=dir+'states_overall_2.png')
plot_episode_data(states_average_1, path=dir+'states_average_1.png')
plot_episode_data(states_average_2, path=dir+'states_average_2.png')

plt.show()


 