import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        num_agents = 3
        num_obstacles = 10
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collidable = True
            agent.collide = False
            agent.silent = True

        # Add obstacles
        world.landmarks = [Landmark() for i in range(num_obstacles)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collidable = True
            landmark.collide = False
            landmark.movable = False

        world.landmarks.append(Landmark())
        world.landmarks[-1].name = 'goal landmark'
        world.landmarks[-1].collide = False
        world.landmarks[-1].collidable = False
        world.landmarks[-1].movable = False

        # make initial conditions
        self.goal_dist = 1.0
        self.reset_world(world)
        return world

    def reset_world(self, world):

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.color[0] += 0.8
            landmark.index = i
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # Set a random goal position
        self.goal_pos = np.random.uniform(-3, +3, world.dim_p)

        # Update our goal landmark for visualization
        world.landmarks[-1].color = np.array([0.0, 0.0, 1.0])
        world.landmarks[-1].p_pos = self.goal_pos

        # set random initial states
        for agent in world.agents:
            agent.color = np.array([0.25,0.25,0.25])
            agent.state.p_pos = np.random.uniform(-3,+3, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # After we set all the inital agent and landmark positions, 
        # make sure we aren't in any collisions already
        for agent in world.agents:
            while(self.is_collision(agent, world)):
                agent.state.p_pos = np.random.uniform(-3,+3, world.dim_p)


    def rel_pos_cost(self, pos1, pos2):
        # Distance is the l1 Norm
        dist = np.linalg.norm(pos1 - pos2)
        # Cost is 0 at specified distance, larger otherwise
        cost = -abs(dist - self.goal_dist)
        return cost

    def is_collision(self, agent, world):
        for entity in world.entities:
            # compute actual distance between entities
            delta_pos = entity.state.p_pos - agent.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity.size + agent.size
            if dist < dist_min and entity != agent and entity.collidable:
                return True
        return False

    def reward(self, agent, world):
        total_cost = 0.0
        # Accumulate costs from each agent to each other agent
        for other in world.agents:
            if agent != other:
                total_cost += self.rel_pos_cost(agent.state.p_pos, other.state.p_pos)

        # distance for cost from goal pos
        dist_from_goal = np.linalg.norm(agent.state.p_pos - self.goal_pos)

        total_cost -= dist_from_goal

        # Chosen kind of arbitrarily, collision cost
        total_cost -= 5000.0 if self.is_collision(agent, world) else 0.0

        # Add a cost for movement
        total_cost -= np.sum(abs(agent.state.p_vel))
        return total_cost

    # Our observation include every agents velocity and our current positions
    def observation(self, agent, world):
        obs = agent.state.p_vel
        obs = np.append(obs, self.goal_dist)
        for other in world.agents:
            if agent != other:
                rel_pos = (agent.state.p_pos - other.state.p_pos)
                obs = np.concatenate([obs, rel_pos, other.state.p_vel])

        for obst in world.landmarks:
            rel_pos = (agent.state.p_pos - obst.state.p_pos)
            obs = np.concatenate([obs, rel_pos])
        rel_goal = agent.state.p_pos - self.goal_pos
        obs = np.concatenate([obs, rel_goal])
        return obs

    def info(self, agent, world):
        return self.goal_pos

    # In this scenario, we only go terminal if there is a collision
    def done(self, agent, world):
        if self.is_collision(agent, world):
            return True
        return False
        

