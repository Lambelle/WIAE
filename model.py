import tensorflow as tf
from tensorflow import keras
from keras import layers


def get_discriminator_model(D_init, nD, nO, nI, k_size):
    d_input = layers.Input(shape=(nI - k_size + 1,))
    x = layers.Dense(nD, activation="tanh", kernel_initializer=D_init)(d_input)
    # x=layers.BatchNormalization()(x)
    x = layers.Dense(nD, activation="tanh", kernel_initializer=D_init)(x)
    # x=layers.BatchNormalization()(x)
    x = layers.Dense(nD, activation="tanh", kernel_initializer=D_init)(x)
    # x=layers.BatchNormalization()(x)
    x = layers.Dense(nO, activation="linear")(x)
    d_model = tf.keras.models.Model(d_input, x, name="discriminator")
    return d_model


def get_generator_model(G_init, nD, nI, k_size):
    g_input = layers.Input(shape=(nI, 1))
    x = layers.Conv1D(
        filters=nD,
        kernel_size=k_size,
        padding="valid",
        activation="tanh",
        kernel_initializer=G_init,
    )(g_input)
    # x=layers.BatchNormalization()(x)
    x = layers.Conv1D(
        filters=nD / 2,
        kernel_size=1,
        padding="valid",
        activation="tanh",
        kernel_initializer=G_init,
    )(x)
    # x=layers.BatchNormalization()(x)
    x = layers.Conv1D(
        filters=nD / 4,
        kernel_size=1,
        padding="valid",
        activation="tanh",
        kernel_initializer=G_init,
    )(x)
    # x=layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=1, kernel_size=1, padding="valid", activation="linear")(x)
    x = tf.reshape(x, [-1, nI - k_size + 1])
    g_model = tf.keras.models.Model(g_input, x, name="generator")
    return g_model


def get_decoder_model(Decoder_init, nI, k_size, nD):
    de_input = layers.Input(shape=(nI - k_size + 1, 1))
    x = layers.Conv1D(
        filters=nD,
        kernel_size=k_size,
        padding="causal",
        activation="tanh",
        kernel_initializer=Decoder_init,
    )(de_input)
    # x=layers.BatchNormalization()(x)
    x = layers.Conv1D(
        filters=nD / 2,
        kernel_size=1,
        padding="valid",
        activation="tanh",
        kernel_initializer=Decoder_init,
    )(x)
    # x=layers.BatchNormalization()(x)
    x = layers.Conv1D(
        filters=nD / 4,
        kernel_size=1,
        padding="valid",
        activation="tanh",
        kernel_initializer=Decoder_init,
    )(x)
    # x=layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=1, kernel_size=1, padding="valid", activation="linear")(x)
    x = tf.reshape(x, [-1, nI - k_size + 1])
    de_model = tf.keras.models.Model(de_input, x, name="decoder")
    return de_model


def get_discriminator_de_model(Ded_init, nI, nD, nO):
    ded_input = layers.Input(shape=(nI - 2 * k_size + 2,))
    x = layers.Dense(nD, activation="tanh", kernel_initializer=Ded_init)(ded_input)
    # x=layers.BatchNormalization()(x)
    x = layers.Dense(nD / 2, activation="tanh", kernel_initializer=Ded_init)(x)
    # x=layers.BatchNormalization()(x)
    x = layers.Dense(nD / 4, activation="tanh", kernel_initializer=Ded_init)(x)
    # x=layers.BatchNormalization()(x)
    x = layers.Dense(nO, activation="linear")(x)
    ded_model = tf.keras.models.Model(ded_input, x, name="discriminator_de")
    return ded_model
