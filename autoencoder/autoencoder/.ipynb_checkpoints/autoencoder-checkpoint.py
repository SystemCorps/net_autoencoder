from keras.layers import Input, Dense
from keras.models import Model

import numpy as np
from sklearn.cross_validation import train_test_split
import random
import matplotlib.pyplot as plt

import os

from glob import glob

img_size = 60*44
encoding_dim = 200

input_img = Input(shape=(img_size,))
encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
decoded = Dense(img_size, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
