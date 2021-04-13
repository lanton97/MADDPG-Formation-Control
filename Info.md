1 - Acceleration and velocity

They are changed inside custom class Scenario(BaseScenario) during the environmnet creation (make_world)

- accel - Acceleration is used to amplify the inputs
standard value is none but later is assigned to a variable called sensitivity initialized as 5
action = action*acceleration

- max_speed is initialized as none but is just a the name says. A maximum allowed value.


## TODO LIST

- ADD best overall and best average model in MADDPG (and maybe DECDDPG)
- DDPG has if done: break commented out but MAPPDG has it working. Which one should stay?
- plot_episode_data is not using states