import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(3)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
        # make initial conditions
        self.goal_dist = 1.0
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def rel_pos_cost(self, pos1, pos2):
        dist = np.linalg.norm(pos1 - pos2)
        cost = -abs((dist - self.goal_dist)/(dist))
        return cost

    def reward(self, agent, world):
        total_cost = 0.0
        for other in world.agents:
            if agent != other:
                total_cost += self.rel_pos_cost(agent.state.p_pos, other.state.p_pos)
        return total_cost

    def observation(self, agent, world):

        return np.concatenate([agent.state.p_vel, agent.state.p_pos])
