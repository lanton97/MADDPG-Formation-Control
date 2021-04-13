import matplotlib.pyplot as plt
from gym import spaces
from envs.cont_environment import ContMultiAgentEnv
import envs.scenarios as scenarios
import os, datetime, imageio
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from matplotlib.ticker import MaxNLocator

# Adapted from the multi-agent particle env to use a continuous environment
def make_env(scenario_name, benchmark=False):

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,  scenario.benchmark_data)
    else:
        env = ContMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=scenario.info, done_callback=scenario.done, shared_viewer=True)
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


def plot_episode_data(scenario, infos, path = []):
    if(scenario == 'simple_formation'):
        plot_episode_data_simple_formation(infos, path)
    elif(scenario == 'formation_w_goal'):
        plot_episode_data_formation_w_goal(infos, path)
   

def plot_episode_data_simple_formation(infos, path = []):
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

def plot_episode_data_formation_w_goal(infos, path = []):
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
    
    fig, axs = plt.subplots(3)  #create figure and axes

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
    axs[0].set_title("Particle Trajectories")
    axs[0].set(xlabel='X', ylabel='Y')
    axs[0].locator_params(axis="both", integer=True, tight=True)

    axs[1].axhline(y=1, color='r', linestyle='--', label='Goal distance')
    axs[1].plot(rels01)
    axs[1].plot(rels12)
    axs[1].plot(rels20)
    axs[1].set(xlabel='Steps', ylabel='Distance')
    axs[1].set_title("Relative distances")
    axs[1].locator_params(axis="both", integer=True, tight=True)

    axs[2].axhline(y=0, color='r', linestyle='--', label='Zero line')
    axs[2].plot(rels01)
    axs[2].set(xlabel='Steps', ylabel='Distance')
    axs[2].set_title("Distance of formation to goal")
    axs[2].locator_params(axis="both", integer=True, tight=True)

    plt.savefig(path)


def save_render(path,images):
    print("Saving Gif!!!")
    with imageio.get_writer(path, mode='I',fps=30) as writer:
        for img in images:
            writer.append_data(img)
    print("Finished saving Gif")



#https://github.com/mrwojo/geometric_median
def geometric_median(points, method='auto', options={}):
    """
    Calculates the geometric median of an array of points.
    method specifies which algorithm to use:
        * 'auto' -- uses a heuristic to pick an algorithm
        * 'minimize' -- scipy.optimize the sum of distances
        * 'weiszfeld' -- Weiszfeld's algorithm
    """

    points = np.asarray(points)

    if len(points.shape) == 1:
        # geometric_median((0, 0)) has too much potential for error.
        # Did the user intend a single 2D point or two scalars?
        # Use np.median if you meant the latter.
        raise ValueError("Expected 2D array")

    if method == 'auto':
        if points.shape[1] > 2:
            # weiszfeld tends to converge faster in higher dimensions
            method = 'weiszfeld'
        else:
            method = 'minimize'

    return _methods[method](points, options)


def minimize_method(points, options={}):
    """
    Geometric median as a convex optimization problem.
    """

    # objective function
    def aggregate_distance(x):
        return cdist([x], points).sum()

    # initial guess: centroid
    centroid = points.mean(axis=0)

    optimize_result = minimize(aggregate_distance, centroid, method='COBYLA')

    return optimize_result.x


def weiszfeld_method(points, options={}):
    """
    Weiszfeld's algorithm as described on Wikipedia.
    """

    default_options = {'maxiter': 1000, 'tol': 1e-7}
    default_options.update(options)
    options = default_options

    def distance_func(x):
        return cdist([x], points)

    # initial guess: centroid
    guess = points.mean(axis=0)

    iters = 0

    while iters < options['maxiter']:
        distances = distance_func(guess).T

        # catch divide by zero
        # TODO: Wikipedia cites how to deal with distance 0
        distances = np.where(distances == 0, 1, distances)

        guess_next = (points/distances).sum(axis=0) / (1./distances).sum(axis=0)

        guess_movement = np.sqrt(((guess - guess_next)**2).sum())

        guess = guess_next

        if guess_movement <= options['tol']:
            break

        iters += 1

    return guess


_methods = {
    'minimize': minimize_method,
    'weiszfeld': weiszfeld_method,
}
