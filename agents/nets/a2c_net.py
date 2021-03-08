import tensorflow as tf
from tensorflow.keras.layers import Dense, Input


def generate_a2c_net(
        obs_shape,
        num_actions
        ):
    state_input = Input(shape=obs_shape)

    dense = Dense(128)(state_input)
    dense = Dense(128)(dense)

    act_mean = Dense(num_actions, activation="linear")(dense)
    act_var  = Dense(num_actions, activation="softplus")(dense)

    critic_val = Dense(1, activation="linear")(dense)

    model = tf.keras.Model([state_input], [act_mean, act_var, critic_val])

    return model

