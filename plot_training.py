from agents import dec_ddpg_runner
from agents import maddpg_runner
from train_and_test.util import *
import argparse


parser = argparse.ArgumentParser(description='File to plot training data.')

parser.add_argument('--avg-files', dest='avg_file_paths', nargs='+', default=[],
                    help='Files which hold the average training data')
parser.add_argument('--avg-labels', dest='avg_labels', nargs='+', default=[],
                    help='Labels for the avergae training data.')

parser.add_argument('--episodic-files', dest='ep_file_paths', nargs='+', default=[],
                    help='Files which hold the episodic training data.')
parser.add_argument('--episodic-labels', dest='ep_labels', nargs='+', default=[],
                    help='Labels for the episodic training data.')


args = parser.parse_args()

plt.axhline(y=0, color='r', linestyle='--', label='Zero Line')

cmap = plt.cm.get_cmap('hsv', len(args.avg_file_paths) + 1)

for i in range(len(args.avg_file_paths)):
    colour = cmap(i)
    avg_data = np.load(args.avg_file_paths[i])
    plt.plot(avg_data, label=args.avg_labels[i], c=colour)

    if len(args.avg_file_paths)==len(args.ep_file_paths):
        ep_data = np.load(args.ep_file_paths[i])
        plt.plot(ep_data, label=args.ep_labels[i], alpha=0.4, c=colour)


plt.legend()
plt.title('Training Curve')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.show()


 
