from agents import dec_ddpg_runner
from agents import maddpg_runner
from train_and_test.util import *
import argparse


parser = argparse.ArgumentParser(description='File to run experiments for som scenario with a centralized agent.')
parser.add_argument('--agent', dest='agent', default='decddpg',
        help='Name of the agent types: Valid values are \'decddpg\' and \'maddpg\'')
parser.add_argument('--scenario', dest='scenario_name', default='simple_formation',
                    help='Name of the scenario we want to run')
parser.add_argument('--num_eps', dest='num_eps', default=1000,
                    help='Number of episodes to train for.', type=int)
parser.add_argument('--save_gifs', dest='gif', default=True,
                    help='True to save gifs, False not to.', type=bool)
parser.add_argument('--save_models', dest='save_model', default=False,
                    help='True to save gifs, False not to.', type=bool)

args = parser.parse_args()

agent = None
dir = generate_path("./experiments/" + args.agent + "/" + args.scenario_name + "")

env = make_env(args.scenario_name)

if args.agent == "decddpg":
    agent = dec_ddpg_runner.DecDDPGRunner(env)
elif args.agent == "maddpg":
    agent = maddpg_runner.MADDPGRunner(env)
else:
    raise Exception("Please provide valid agent type")

num_episodes = args.num_eps
num_steps = 300

print("Training Model")
rewards, avg_rewards, info = agent.train(num_episodes=num_episodes, num_steps=num_steps)

if args.save_model:
    agent.save_agents(suffix=args.scenario_name)

input("Press to run trained model") # Even requesting a key press, for us to be prepared to watch the model
states_last_1, episodic_reward_last_1, info_last_1, images_last_1 = agent.run_episode(num_steps, waitTime = 0.0, mode='rgb_array') # Should we store the model that obtained the best reward? or always use the last one?
states_last_2, episodic_reward_last_2, info_last_2, images_last_2 = agent.run_episode(num_steps, waitTime = 0.0, mode='rgb_array')  # Should we store the model that obtained the best reward? or always use the last one?
states_overall_1, episodic_reward_overall_1, info_last_overall_1, images_overall_1 = agent.run_episode(num_steps, waitTime = 0, mode='rgb_array', policy_param='best_overall') # Should we store the model that obtained the best reward? or always use the last one?
states_overall_2, episodic_reward_overall_2, info_last_overall_2, images_overall_2 = agent.run_episode(num_steps, waitTime = 0, mode='rgb_array', policy_param='best_overall') # Should we store the model that obtained the best reward? or always use the last one?
states_average_1, episodic_reward_average_1, info_last_average_1, images_average_1 = agent.run_episode(num_steps, waitTime = 0, mode='rgb_array', policy_param='best_average') # Should we store the model that obtained the best reward? or always use the last one?
states_average_2, episodic_reward_average_2, info_last_average_2, images_average_2 = agent.run_episode(num_steps, waitTime = 0, mode='rgb_array', policy_param='best_average') # Should we store the model that obtained the best reward? or always use the last one?
env.close()

if args.gif:
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


 
