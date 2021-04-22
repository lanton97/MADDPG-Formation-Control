from agents import dec_ddpg_runner
from agents import maddpg_runner
from train_and_test.util import *
import argparse

parser = argparse.ArgumentParser(description='File to plot training data.')

parser.add_argument('--avg-files', dest='avg_file_paths', nargs='+', default=["experiments/maddpg/formation_w_coll_avoidance/report/2021-04-21_21-48-36/avg_rewards.npy", "experiments/ddpg/formation_w_coll_avoidance/report/2021-04-21_21-49-28/avg_rewards.npy", "experiments/decddpg/formation_w_coll_avoidance/report/2021-04-22_00-30-17/avg_rewards.npy"],
                    help='Files which hold the average training data')
parser.add_argument('--labels', dest='labels', nargs='+', default=["MADDPG", "DDPG", "Dec-DDPG" ],
                    help='Labels for the training data.')

parser.add_argument('--episodic-files', dest='ep_file_paths', nargs='+', default=["experiments/maddpg/formation_w_coll_avoidance/report/2021-04-21_21-48-36/rewards.npy", "experiments/ddpg/formation_w_coll_avoidance/report/2021-04-21_21-49-28/rewards.npy", "experiments/decddpg/formation_w_coll_avoidance/report/2021-04-22_00-30-17/rewards.npy"],
                    help='Files which hold the episodic training data.')


args = parser.parse_args()

plt.axhline(y=0, color='r', linestyle='--')

cmap = plt.cm.get_cmap('hsv', len(args.avg_file_paths) + 1)

for i in range(len(args.avg_file_paths)):
    colour = cmap(i)
    avg_data = np.load(args.avg_file_paths[i])
    plt.plot(avg_data, label=args.labels[i], c=colour)

    if len(args.avg_file_paths)==len(args.ep_file_paths):
        ep_data = np.load(args.ep_file_paths[i])
        plt.plot(ep_data, alpha=0.4, c=colour)


plt.legend(loc='lower right')
plt.title('Training Curve')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.show()


 
