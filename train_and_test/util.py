import matplotlib.pyplot as plt
from gym import spaces
from envs.cont_environment import ContMultiAgentEnv
import envs.scenarios as scenarios
import os, datetime, imageio
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

