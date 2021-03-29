import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        num_agents = 3
        num_obstacles = 6
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True

        # Add obstacles
        world.landmarks = [Landmark() for i in range(num_obstacles)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False

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
        self.goal_pos = np.random.uniform(-1, +1, world.dim_p)

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def rel_pos_cost(self, pos1, pos2):
        # Distance is the l1 Norm
        dist = np.linalg.norm(pos1 - pos2)
        # Cost is 0 at specified distance, larger otherwise
        cost = max(-abs((dist - self.goal_dist)/(dist)), -1.0)
        return cost

    def is_collision(self, agent, world):
        for a,entity_a in enumerate(world.entities):
           [f_a, f_b] = world.get_collision_force(entity_a, agent)
           if(f_a is not None or f_b is not None):
               return True

        return False

    def reward(self, agent, world):
        total_cost = 0.0
        # Accumulate costs from each agent to each other agent
        for other in world.agents:
            if agent != other:
                total_cost += self.rel_pos_cost(agent.state.p_pos, other.state.p_pos)

        # Squared distance for cost from goal pos
        dist_from_goal = np.square(np.linalg.norm(agent.state.p_pos - self.goal_pos))

        total_cost -= dist_from_goal

        # Chosen kind of arbitrarily, collision cost
        total_cost -= 2000.0 if self.is_collision(agent, world) else 0.0

        # Add a cost for movement
        total_cost -= np.sum(abs(agent.state.p_vel))
        return total_cost

    # Our observation include every agents velocity and out current positions
    def observation(self, agent, world):
        obs = agent.state.p_vel
        for other in world.agents:
            if agent != other:
                rel_pos = (agent.state.p_pos - other.state.p_pos)
                obs = np.concatenate([obs, rel_pos, agent.state.p_vel])

        for obst in world.landmarks:
            rel_pos = (agent.state.p_pos - obst.state.p_pos)
            obs = np.concatenate([obs, rel_pos])
        rel_goal = agent.state.p_pos - self.goal_pos
        obs = np.concatenate([obs, rel_goal])
        return obs

    # In this scenario, we only go terminal if there is a collision
    def done(self, agent, world):
        if self.is_collision(agent, world):
            return True
        return False
        

