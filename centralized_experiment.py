from agents import ddpg
from envs.centralized_env import CentralizedEnvWrapper
from train_and_test.util import *
import argparse

parser = argparse.ArgumentParser(description='File to run experiments for som scenario with a centralized agent.')
parser.add_argument('--scenario', dest='scenario_name', default='formation_w_coll_avoidance',
                    help='Name of the scenario we want to run: formation_w_coll_avoidance, formation_w_goal or simple_formation')

parser.add_argument('--num_eps', dest='num_eps', default=1000,
                    help='Number of episodes to train for.', type=int)

parser.add_argument('--save_images', dest='images', default='True',
                    help='True to save images and gifs, anything else not to.')

parser.add_argument('--save_models', dest='save_model', default='False',
                    help='True to save models, anything not to.')

parser.add_argument('--load_models', dest='load_model', default='True',
                    help='True to load models, anything else not to.')

parser.add_argument('--train', dest='train', default='False',
                    help='True to train models, anything else not to.')

parser.add_argument('--save_suffix', dest='save_suffix', default="report",
                    help='Suffix for saving the file.')
                    
parser.add_argument('--load_suffix', dest='load_suffix', default="report",
                    help='Suffix for loading the file.')


args = parser.parse_args()

# Start Experiment
# Experiment path
if args.images=='True':
    dir = generate_path("./experiments/" + "/ddpg/" + args.scenario_name + "/" +  args.save_suffix)

# Create the DDPG agent and the centralized environment
env = CentralizedEnvWrapper(make_env(args.scenario_name))
agent = ddpg.DDPGAgent(env)

# Load a trained model with the given path
if args.load_model=='True':
    agent.load_models(suffix=args.scenario_name + args.load_suffix)

num_episodes = args.num_eps
num_steps = 300

# Train the agent
if args.train=='True':
    print("Training Model")
    rewards, avg_rewards, info = agent.train(num_episodes=num_episodes, num_steps=num_steps)

# Save the model if specified
if args.save_model=='True':
    agent.save_models(suffix=args.scenario_name+args.save_suffix)

# Run the agents with the last policy, the best overall policy, and the best average policy based on episodic reward
states_last_1, episodic_reward_last_1, info_last_1, images_last_1 = agent.run_episode(num_steps, waitTime = 0,mode='rgb_array') 
states_last_2, episodic_reward_last_2, info_last_2, images_last_2 = agent.run_episode(num_steps, waitTime = 0,mode='rgb_array') 
states_overall_1, episodic_reward_overall_1, info_last_overall_1, images_overall_1 = agent.run_episode(num_steps, waitTime = 0, mode='rgb_array', policy_param='best_overall')
states_overall_2, episodic_reward_overall_2, info_last_overall_2, images_overall_2 = agent.run_episode(num_steps, waitTime = 0, mode='rgb_array', policy_param='best_overall')
states_average_1, episodic_reward_average_1, info_last_average_1, images_average_1 = agent.run_episode(num_steps, waitTime = 0, mode='rgb_array', policy_param='best_average')
states_average_2, episodic_reward_average_2, info_last_average_2, images_average_2 = agent.run_episode(num_steps, waitTime = 0, mode='rgb_array', policy_param='best_average')
env.close()

# Save the gifs from our runs
if args.images=='True':
    save_render(dir + "/images_last_1.gif", images_last_1)
    save_render(dir + "/images_last_2.gif", images_last_2)
    save_render(dir + "/images_overall_1.gif", images_overall_1)
    save_render(dir + "/images_overall_2.gif", images_overall_2)
    save_render(dir + "/images_average_1.gif", images_average_1)
    save_render(dir + "/images_average_2.gif", images_average_2)

# Plot training data
if ((args.train=='True') & (args.images=='True')):
    np.save(dir+'rewards', rewards)
    np.save(dir+'avg_rewards', avg_rewards)
    plot_train_data(rewards, avg_rewards, path=dir+'train_data.png')

# Plot information on the agent behaviour
if args.images=='True':
    plot_episode_data(args.scenario_name, info_last_1, path=dir, file_name = 'info_last_1')
    plot_episode_data(args.scenario_name, info_last_2, path=dir, file_name = 'info_last_2')
    plot_episode_data(args.scenario_name, info_last_overall_1, path=dir, file_name = 'info_overall_1')
    plot_episode_data(args.scenario_name, info_last_overall_2, path=dir, file_name = 'info_overall_2')
    plot_episode_data(args.scenario_name, info_last_average_1, path=dir, file_name = 'info_average_1')
    plot_episode_data(args.scenario_name, info_last_average_2, path=dir, file_name = 'info_average_2')

plt.show()


 
