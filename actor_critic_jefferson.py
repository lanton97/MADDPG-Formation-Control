import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Softmax
import matplotlib.pyplot as plt


# Value with two hidden layers and a 1-dimensional linear output
# Will be instantiated as 2-input NN
class ValueModel(Model):
    def __init__(self, units_1, units_2):
        super(ValueModel,self).__init__()
        self.layer_1 = tf.keras.layers.Dense(units_1, activation='relu',name='Hidden 1')
        self.layer_2 = tf.keras.layers.Dense(units_2, activation='relu',name='Hidden 2')
        self.V = tf.keras.layers.Dense(units=1,use_bias=False,name='Value output')

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        return self.V(x)

# Policy with two hidden layers and 2 one-dimensional outputs
# A linear output for mean and a softmax output for standard deviation
# Will be instantiated as 2-input NN
class PolicyModel(Model):
    def __init__(self, units_1, units_2):
        super(PolicyModel,self).__init__()
        self.layer_1 = tf.keras.layers.Dense(units_1, activation='relu',name='Hidden 1')
        self.layer_2 = tf.keras.layers.Dense(units_2, activation='relu',name='Hidden 2')
        self.mean = tf.keras.layers.Dense(units=1, use_bias=False,name='Mean output')
        self.std = tf.keras.layers.Dense(units=1, activation='softplus', use_bias=False,name='std output')

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        mean_out = self.mean(x)
        std_out = self.std(x)
        return mean_out, std_out


# Loss definitions (TODO: test reduce mean vs reduce sum)
def CriticLoss(Q, value):
    return tf.reduce_mean(tf.square(Q-value)) 

def ActorLoss(advantages, log_probs, entropy):
    return -tf.reduce_mean(advantages*log_probs + 0.1*entropy)


def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
  plt.plot(running_avg)
  plt.title("Running Average")
  #plt.show()

def main():

    seedn = 0
    np.random.seed(seedn)
    tf.random.set_seed(seedn)
    # Instantiating policy model
    criticModel = ValueModel(400,400)
    print(criticModel(tf.ones((1, 2))))
    criticModel.summary()

    # Instantiating policy model
    actorModel = PolicyModel(40,40)
    print(actorModel(tf.ones((1, 2))))
    actorModel.summary()

    # Optmizer selection
    opt_actor = tf.keras.optimizers.SGD(learning_rate=0.001)
    opt_critic = tf.keras.optimizers.SGD(learning_rate=0.001)
    #Learning parameters
    num_episodes = 250
    gamma = 0.99

    # Setup environment
    env = gym.make('MountainCarContinuous-v0')

    #TODO: Scale state space https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
    episode_history = []
    for episode in range(num_episodes):
        state = env.reset()   # state.shape -> (2,)
        state = state.reshape(1,2)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        reward_total = 0
        steps = 0
        done = False
        
        while (not done):
            #env.render()          
            #action = tf.constant(0.0)
            with tf.GradientTape(persistent=True) as tape:

                value = criticModel(state)
                # Actor - Sample action
                mean, std = actorModel(state)   # TODO: test predict instead of model
                norm_dist = tfp.distributions.Normal(loc=mean, scale=std)
                
                action = tf.clip_by_value(norm_dist.sample(seed=seedn), 
                                         env.action_space.low[0], 
                                         env.action_space.high[0]) #action.shape = (1,1)
                # Apply action to environment
                state_next, reward, done, info = env.step(action)
                state_next = state_next.reshape(1,2)
                #state_next = tf.convert_to_tensor(state_next, dtype=tf.float32)
                value_next = criticModel(state_next)

                Q = reward + gamma*value_next

                criticLoss = tf.reduce_mean(tf.square(Q-value)) 
                #criticGrads = tape.gradient(criticLoss, criticModel.trainable_variables)


                log_probs = norm_dist.log_prob(action + 1e-5)
                actorLoss = -tf.reduce_mean((Q-value)*log_probs + 0.1*norm_dist.entropy())
                #actorGrads = tape.gradient(actorLoss, actorModel.trainable_variables)

            criticGrads = tape.gradient(criticLoss, criticModel.trainable_variables)
            actorGrads = tape.gradient(actorLoss, actorModel.trainable_variables)
             # Apply gradient
            opt_actor.apply_gradients(zip(actorGrads, actorModel.trainable_variables))
            opt_critic.apply_gradients(zip(criticGrads, criticModel.trainable_variables))

            steps +=1
            reward_total += reward
            state = state_next
        episode_history.append(reward_total)
        print("Episode: {}, Steps in episode: {}, Cumulative reward: {}".format(episode, steps, reward_total))
    #env.close()
    plt.figure(1)
    plt.plot(episode_history)
    plt.title("Rewards")
    plt.figure(2)
    plot_running_avg(episode_history)
    plt.show()

if __name__ == '__main__':
  main()