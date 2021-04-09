import matplotlib.pyplot as plt
from gym import spaces
from envs.cont_environment import ContMultiAgentEnv
import envs.scenarios as scenarios
import os, datetime, imageio
import numpy as np
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
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done, shared_viewer=True)
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

def plot_episode_data(states, infos, path = []):
    rels01 = []
    rels12 = []
    rels20 = []
    #TODO: Generalize code
    for info in infos:
        rel01 = np.sqrt(np.sum((info[0] - info[1])**2))
        rel12 = np.sqrt(np.sum((info[1] - info[2])**2))
        rel20 = np.sqrt(np.sum((info[2] - info[0])**2))
        rels01.append(rel01)
        rels12.append(rel12)
        rels20.append(rel20)
    
    fig, axs = plt.subplots(2)  #create figure and axes

    for i in range(len(infos[0])):
       pos =  [row[i] for row in infos]
       x = [row[0] for row in pos]
       y = [row[1] for row in pos]
       axs[0].plot(x, y)
       axs[0].plot(x[-1], y[-1], 'or')
    #plt.xlim((-3,3))
    #plt.ylim((-3,3))
    axs[0].axis('equal')
    axs[0].set(xlabel='X', ylabel='Y')
    plt.xlabel('X')
    plt.ylabel('Y')

    axs[1].plot(rels01)
    axs[1].plot(rels12)
    axs[1].plot(rels20)
    axs[1].set(xlabel='Steps', ylabel='Relative distances')
    fig.suptitle("Particle Trajectory")
    plt.savefig(path)

def save_render(path,images):
    print("Saving Gif!!!")
    with imageio.get_writer(path, mode='I',fps=30) as writer:
        for img in images:
            writer.append_data(img)
    print("Finished saving Gif")

