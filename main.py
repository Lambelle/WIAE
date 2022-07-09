import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import csv
import matplotlib.pyplot as plt
import pandas as pd
import model


inp = []
with open("/content/drive/My Drive/Colab Notebooks/Relaxed IAE/MC.txt") as csvfile:
    reader = csv.reader(
        csvfile, quoting=csv.QUOTE_NONNUMERIC
    )  # change contents to floats
    for row in reader:  # each row is a list
        inp.append(row)
    GenSummer = np.asarray(inp)
    print(GenSummer.shape)
dataSet = GenSummer.transpose()
dataSize = GenSummer.shape[0]
print(dataSet.shape)


nO = 1
nD = 100
k_size = 20
nI = 50

D_init = tf.keras.initializers.he_normal(seed=1)
D = model.get_discriminator_model(D_init, nD, nO, nI, k_size)
D.summary()

G_init = tf.keras.initializers.he_normal(seed=0)
G = model.get_generator_model(G_init, nD, nI, k_size)
G.summary()


Decoder_init = tf.keras.initializers.he_normal(seed=0)
De = model.get_decoder_model(Decoder_init, nI, k_size, nD)
De.summary()

Ded_init = tf.keras.initializers.he_normal(seed=1)
DeD = model.get_discriminator_de_model(Ded_init, nI, nD, nO, k_size)
DeD.summary()
