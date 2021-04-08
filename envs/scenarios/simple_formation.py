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
            #agent.max_speed = 1.0 # None
            #agent.accel = 0.5#standard is none but later used as 5
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
        # Distance is the l1 Norm
        dist = np.linalg.norm(pos1 - pos2)
        # Cost is 0 at specified distance, larger otherwise
        cost = max(-abs((dist - self.goal_dist)/(dist)), -1.0)
        return cost

    def reward(self, agent, world):
        total_cost = 0.0
        # Accumulate costs from each agent to each other agent
        for other in world.agents:
            if agent != other:
                total_cost += self.rel_pos_cost(agent.state.p_pos, other.state.p_pos)

        # Add a cost for movement
        total_cost -= np.sum(abs(agent.state.p_vel))
        return total_cost

    # Our observation include every agents velocity and out current positions
    def observation(self, agent, world):
        rel_pos = []
        other_vels = []
        for other in world.agents:
            if agent != other:
                rel_pos.append(agent.state.p_pos - other.state.p_pos)
                other_vels.append(other.state.p_vel)
        return np.append(np.concatenate([agent.state.p_vel, rel_pos[0], other_vels[0], rel_pos[1], other_vels[1]]), self.goal_dist)

    def done(self, agent, world):
        return False
