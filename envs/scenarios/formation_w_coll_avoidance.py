import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        num_agents = 3
        num_obstacles = 7

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collidable = True
            agent.collide = False
            agent.silent = True
            agent.max_speed = 1.0 # None
            agent.accel = 2.0#standard is none but later used as 5

        # Add obstacles
        world.obstacles = [Landmark() for i in range(num_obstacles)]
        for i, landmark in enumerate(world.obstacles):
            landmark.name = 'obstacle %d' % i
            landmark.collidable = True
            landmark.collide = False
            landmark.movable = False

        world.goal = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.goal):
            landmark.name = 'goal landmark %d' % i
            landmark.collide = False
            landmark.collidable = False
            landmark.movable = False

        world.landmarks = world.goal   #TODO: copy goal to landmark instead of by reference (make sure it doesnt cause any bugs)
        world.landmarks += world.obstacles
        
        # make initial conditions
        self.goal_dist = 1.0
        self.reset_world(world)
        return world

    def reset_world(self, world):

        for i, landmark in enumerate(world.obstacles):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.color[0] += 0.8
            landmark.index = i
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # Set a random goal position
        self.goal_pos = np.random.uniform(-1, +1, world.dim_p)

        # Update our goal landmark for visualization
        world.goal[0].color = np.array([0.0, 0.0, 1.0])
        world.goal[0].state.p_pos = self.goal_pos
        world.goal[0].state.p_vel = np.zeros(world.dim_p)

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
        for entity in (world.agents + world.obstacles + world.goal):
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
        formation_weight = 10
        # Accumulate costs from each agent to each other agent
        for other in world.agents:
            if agent != other:
                total_cost += formation_weight*self.rel_pos_cost(agent.state.p_pos, other.state.p_pos)

        # distance for cost from goal pos
        dist_from_goal = 10*np.linalg.norm(agent.state.p_pos - self.goal_pos)

        total_cost -= dist_from_goal

        # Chosen kind of arbitrarily, collision cost
        total_cost -= 10 if self.is_collision(agent, world) else 0.0

        return total_cost

    # Our observation include every agents velocity and our current positions
    def observation(self, agent, world):
        obs = agent.state.p_vel
        obs = np.append(obs, self.goal_dist)
        for other in world.agents:
            if agent != other:
                rel_pos = (agent.state.p_pos - other.state.p_pos)
                obs = np.concatenate([obs, rel_pos, other.state.p_vel])

        for obst in world.obstacles:
            rel_pos = (agent.state.p_pos - obst.state.p_pos)
            obs = np.concatenate([obs, rel_pos])
        rel_goal = agent.state.p_pos - self.goal_pos
        obs = np.concatenate([obs, rel_goal])
        return obs

    def info(self, agent, world):
        return [self.goal_pos, world.obstacles]

    def done(self, agent, world):
        return False
        

