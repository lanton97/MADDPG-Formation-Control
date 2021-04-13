import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        num_agents = 3
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.max_speed = 1.0 # None
            agent.accel = 2.0#standard is none but later used as 5

        # Add obstacles
        world.landmarks = [Landmark()]

        # make initial conditions
        self.goal_dist = 1.0
        self.reset_world(world)
        return world

    def reset_world(self, world):

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.color[1] += 0.8
            landmark.index = i

        # Set a random goal position
        self.goal_pos = np.random.uniform(-3, +3, world.dim_p)

        world.landmarks[0].state.p_pos = self.goal_pos
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

        # set random initial states
        for agent in world.agents:
            agent.color = np.array([0.25,0.25,0.25])
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)


    def rel_pos_cost(self, pos1, pos2):
        # Distance is the l1 Norm
        dist = np.linalg.norm(pos1 - pos2)
        # Cost is 0 at specified distance, larger otherwise
        cost = -abs(dist-self.goal_dist)#max(-abs((dist - self.goal_dist)/(dist)), -1.0)
        return cost

    def reward(self, agent, world):
        total_cost = 0.0
        # Accumulate costs from each agent to each other agent
        for other in world.agents:
            if agent != other:
                total_cost += self.rel_pos_cost(agent.state.p_pos, other.state.p_pos)

        # distance for cost from goal pos
        dist_from_goal = np.linalg.norm(agent.state.p_pos - self.goal_pos)

        total_cost -= dist_from_goal

        # Add a cost for movement
        #total_cost -= np.sum(abs(agent.state.p_vel))
        return total_cost

    # Our observation include every agents velocity and our current positions
    def observation(self, agent, world):
        obs = agent.state.p_vel
        obs = np.append(obs, self.goal_dist)
        for other in world.agents:
            if agent != other:
                rel_pos = (agent.state.p_pos - other.state.p_pos)
                obs = np.concatenate([obs, rel_pos, other.state.p_vel])

        rel_goal = agent.state.p_pos - self.goal_pos
        obs = np.concatenate([obs, rel_goal])

        return obs

    def info(self,agent, world):
        return self.goal_pos

    # In this scenario, we only go terminal if there is a collision
    def done(self, agent, world):
        return False
        

