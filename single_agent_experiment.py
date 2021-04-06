from agents import ddpg
from envs.centralized_env import CentralizedEnvWrapper
import multiagent.scenarios as scenarios
from train_and_test.util import *

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


 
