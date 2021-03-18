import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, BatchNormalization, Concatenate
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape, Lambda
from tensorflow.keras.layers import TimeDistributed

# A very simple actor network implementations for testing
def generate_critic_network(
        num_states,
        num_actions
        ):

    initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

    # State as input
    state_input = tf.keras.layers.Input(shape=(num_states))
    state_out = tf.keras.layers.Dense(16, activation="relu")(state_input)
    state_out = tf.keras.layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = tf.keras.layers.Input(shape=(num_actions))
    action_out = tf.keras.layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = tf.keras.layers.Concatenate()([state_out, action_out])

    out = tf.keras.layers.Dense(128, activation="relu")(concat)
    out = tf.keras.layers.Dense(128, activation="relu")(out)
    outputs = tf.keras.layers.Dense(1)(out)

    # Outputs single value for given state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def generate_baseline_critic_network(
        num_float_states,
        shape_img_states,
        num_actions
        ):

    initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

    # State as input
    state_float_input = Input(shape=num_float_states)
    state_out = Dense(16, activation="relu")(state_float_input)
    state_out = Dense(32, activation="relu")(state_out)

    # For a baseline, just use a dense network
    state_img_input = Input(shape=(shape_img_states))
    flatten = Flatten()(state_img_input)
    img_state_out   = Dense(32)(flatten)

    # Action as input
    action_input = Input(shape=(num_actions))
    action_out = Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = Concatenate()([state_out, img_state_out, action_out])

    out = Dense(64, activation="relu")(concat)
    out = Dense(64, activation="relu")(out)
    outputs = Dense(1)(out)

    # Outputs single value for given state-action
    model = tf.keras.Model([state_float_input, state_img_input,  action_input], outputs)

    return model



def generate_cnnlstm_critic_network(
        num_float_states,
        shape_img_states,
        num_actions
        ):

    initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

    # State as input
    state_float_input = Input(shape=num_float_states)
    state_out = Dense(16, activation="relu")(state_float_input)
    state_out = Dense(32, activation="relu")(state_out)

    # CNNLSTM for the image input
    # CNN Portion First
    state_img_input = Input(shape=(shape_img_states))
    conv_2d         = TimeDistributed(Convolution2D(32, (3, 3))) (state_img_input)
    max_pool        = TimeDistributed(MaxPooling2D(pool_size=(2,2))) (conv_2d)
    #conv_2d         = TimeDistributed(Convolution2D(32, (3, 3)))(max_pool)
    #max_pool        = TimeDistributed(MaxPooling2D(pool_size=(2,2))) (conv_2d)
    flatten         = TimeDistributed(Flatten())(max_pool)
    # LSTM Portion Now
    lstm            = LSTM(units=32)(flatten)
    #lstm            = LSTM(units=64)(lstm)
    img_state_out   = Dense(32)(lstm)

    # Action as input
    action_input = Input(shape=(num_actions))
    action_out = Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = Concatenate()([state_out, img_state_out, action_out])

    out = Dense(64, activation="relu")(concat)
    out = Dense(64, activation="relu")(out)
    outputs = Dense(1)(out)

    # Outputs single value for given state-action
    model = tf.keras.Model([state_float_input, state_img_input,  action_input], outputs)

    return model

