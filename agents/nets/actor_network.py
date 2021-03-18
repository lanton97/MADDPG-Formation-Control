import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, BatchNormalization, Concatenate
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape, Lambda
from tensorflow.keras.layers import TimeDistributed

# A very simple actor network implementations for testing
def generate_actor_network(
        num_states,
        num_actions,
        actions_max
        ):

    initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    inputs = Input(shape=(num_states,))

    out = Dense(256, activation="relu")(inputs)
    out = Dense(256, activation="relu")(out)
    outputs = Dense(num_actions, activation="tanh", kernel_initializer=initializer)(out)
    # Assumes the actions are equal in each direction
    outputs = outputs * actions_max
    model = tf.keras.Model(inputs, outputs)

    return model

def generate_baseline_actor_network(
        num_float_states,
        shape_img_states,
        num_actions,
        actions_max
        ):

    # Initialize our output state to not make our gradient immediately disappear or explode
    initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

    # Float state network - A simple dense network
    float_input     = Input(shape=num_float_states)
    dense           = Dense(64)(float_input)
    #dense           = Dense(64)(dense)
    float_state_out = Dense(32)(dense)

    # We treat the inputs like a collection of floats
    img_input      = Input(shape=shape_img_states)
    flatten      = Flatten()(img_input)
    img_state_out  = Dense(32)(flatten)

    concat = Concatenate()([float_state_out, img_state_out])

    out = Dense(64, activation="relu")(concat)
    out = Dense(64, activation="relu")(out)
    outputs = Dense(num_actions, activation="tanh", kernel_initializer=initializer)(out)
    # Assumes the actions are equal in each direction
    outputs = outputs * actions_max
    model = tf.keras.Model([float_input, img_input], outputs)

    return model


# A very simple actor network implementations for testing
def generate_cnnlstm_actor_network(
        num_float_states,
        shape_img_states,
        num_actions,
        actions_max
        ):

    # Initialize our output state to not make our gradient immediately disappear or explode
    initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

    # Float state network - A simple dense network
    float_input     = Input(shape=num_float_states)
    dense           = Dense(64)(float_input)
    #dense           = Dense(64)(dense)
    float_state_out = Dense(32)(dense)

    # CNNLSTM for the image input
    # CNN Portion First
    img_input      = Input(shape=shape_img_states)
    conv_2d        = TimeDistributed(Convolution2D(3, (3, 3))) (img_input)
    max_pool       = TimeDistributed(MaxPooling2D(pool_size=(2,2))) (conv_2d)
    #conv_2d        = TimeDistributed(Convolution2D(3, (3, 3))) (max_pool)
    #max_pool       = TimeDistributed(MaxPooling2D(pool_size=(2,2))) (conv_2d)
    flatten        = TimeDistributed(Flatten())(max_pool)
    # LSTM Portion Now
    lstm           = LSTM(units=32)(flatten)
    #lstm           = LSTM(units=64)(lstm)
    img_state_out  = Dense(32)(lstm)

    concat = Concatenate()([float_state_out, img_state_out])

    out = Dense(64, activation="relu")(concat)
    out = Dense(64, activation="relu")(out)
    outputs = Dense(num_actions, activation="tanh", kernel_initializer=initializer)(out)
    # Assumes the actions are equal in each direction
    outputs = outputs * actions_max
    model = tf.keras.Model([float_input, img_input], outputs)

    return model

