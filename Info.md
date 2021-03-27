1 - Acceleration and velocity

They are changed inside sustom class Scenario(BaseScenario) during the environmnet creation (make_world)

- accel - Acceleration is used to amplify the inputs
standard value is none but later is assigned to a variable called sensitivity initialized as 5
action = action*acceleration

- max_speed is initialized as none but is just a the name says. A maximum allowed value.



2 - Coordinate system
- The origin in the simple scenario is the agent perspective. This is annoying. I am trying to find a solution for that.
The landmark can be seen around (-1,1) of the agent.
Down arrow moves the veicle up (The environment look like it is going downwards)
Right arrow moves the veicle left (The environment look like it is going towards the right direction)
         y+
           / \
            |
            |
            | 
 --------------------> x+
            |
            |
            |

3 - The simple environment was being loaded from the multiagent library and not from the envs folder