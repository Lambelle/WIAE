from matplotlib.cbook import to_filehandle
import tensorflow as tf


def get_discriminator_model(D_init, nD, nO, nI, k_size):
    d_input = tf.keras.layers.Input(shape=(nI - k_size + 1,))
    x = tf.keras.layers.Dense(nD, activation="tanh")(d_input)
    # x=layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(nD, activation="tanh")(x)
    # x=layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(nD, activation="tanh")(x)
    # x=layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(nO, activation="linear")(x)
    d_model = tf.keras.models.Model(d_input, x, name="discriminator")
    return d_model


def get_generator_model(G_init, nD, nI, k_size):
    g_input = tf.keras.layers.Input(shape=(nI, 1))
    x = tf.keras.layers.Conv1D(
        filters=nD,
        kernel_size=k_size,
        padding="valid",
        activation="tanh",
    )(g_input)
    # x=layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(
        filters=int(nD / 2),
        kernel_size=1,
        padding="valid",
        activation="tanh",
    )(x)
    # x=layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(
        filters=int(nD / 4),
        kernel_size=1,
        padding="valid",
        activation="tanh",
    )(x)
    # x=layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(filters=1, kernel_size=1, padding="valid", activation="linear")(x)
    x = tf.reshape(x, [-1, nI - k_size + 1])
    g_model = tf.keras.models.Model(g_input, x, name="generator")
    return g_model


def get_decoder_model(Decoder_init, nI, k_size, nD):
    de_input = tf.keras.layers.Input(shape=(nI - k_size + 1, 1))
    x = tf.keras.layers.Conv1D(
        filters=nD,
        kernel_size=k_size,
        padding="causal",
        activation="tanh",
        kernel_initializer=Decoder_init,
    )(de_input)
    # x=layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(
        filters=int(nD / 2),
        kernel_size=1,
        padding="valid",
        activation="tanh",
    )(x)
    # x=layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(
        filters=int(nD / 4),
        kernel_size=1,
        padding="valid",
        activation="tanh",
    )(x)
    # x=layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(filters=1, kernel_size=1, padding="valid", activation="linear")(x)
    x = tf.reshape(x, [-1, nI - k_size + 1])
    de_model = tf.keras.models.Model(de_input, x, name="decoder")
    return de_model


def get_discriminator_de_model(Ded_init, nI, nD, nO, k_size):
    ded_input = tf.keras.layers.Input(shape=(nI - 2 * k_size + 2,))
    x = tf.keras.layers.Dense(nD, activation="tanh")(ded_input)
    # x=layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(int(nD / 2), activation="tanh")(x)
    # x=layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(int(nD / 4), activation="tanh")(x)
    # x=layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(nO, activation="linear")(x)
    ded_model = tf.keras.models.Model(ded_input, x, name="discriminator_de")
    return ded_model



