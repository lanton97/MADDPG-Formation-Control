import gym
from agents.a2c_agent import *

env = gym.make("Pendulum-v0")
agent = A2C_agent(env)

rewards = agent.train(1500)
