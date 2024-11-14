import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.python.keras.regularizers import l1

# Configuration variables
HLactivation = 'relu'               # Activation for hidden layers
outputactivation = 'sigmoid'        # Activation for output layer
filtersize = 5                      # Variable kernel size for Conv1D layers
use_l1_regularization = True        # Toggle L1 regularization on or off
l1_lambda = 0.001                   # L1 regularization coefficient, if applicable

# Loss weights
weighted_loss_type = 'exponential'  # Options: 'exponential' or 'square'
weight_min = 100                  # Minimal freq_bin index to weight. Avoids weighting the near zero "peak".

# Channel numbers for each Conv1D layer
channel_num1 = 32
channel_num2 = 16
channel_num3 = 8

# Max pooling down/upsampling ratio
MP_ratio = 4

# Define a weighted loss function based on the variable choice
def weighted_mse(y_true, y_pred):
    if weighted_loss_type == 'exponential':
        weights = tf.exp(y_true)
    elif weighted_loss_type == 'square':
        weights = tf.square(y_true)
    else:
        raise ValueError("Invalid weighted_loss_type. Choose 'exponential' or 'square'.")
    # If we're below the cutoff, just replace the weight with the lowest calculated weight.
    # This way we don't care much about the universal near zero "peak"
    weights[0:weight_min] = np.min(weights)
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))

# Define the encoder
input_data = Input(shape=(1000, 1))

# Optional L1 regularization
if use_l1_regularization:
    regularizer = l1(l1_lambda)
else:
    regularizer = None

# Encoder with variable channel numbers
x = Conv1D(channel_num1, filtersize, activation=HLactivation, padding='same', kernel_regularizer=regularizer)(input_data)
x = MaxPooling1D(MP_ratio, padding='same')(x)
x = Conv1D(channel_num2, filtersize, activation=HLactivation, padding='same', kernel_regularizer=regularizer)(x)
x = MaxPooling1D(MP_ratio, padding='same')(x)
x = Conv1D(channel_num3, filtersize, activation=HLactivation, padding='same', kernel_regularizer=regularizer)(x)
encoded = MaxPooling1D(MP_ratio, padding='same')(x)

# Decoder with variable channel numbers
x = Conv1D(channel_num3, filtersize, activation=HLactivation, padding='same', kernel_regularizer=regularizer)(encoded)
x = UpSampling1D(MP_ratio)(x)
x = Conv1D(channel_num2, filtersize, activation=HLactivation, padding='same', kernel_regularizer=regularizer)(x)
x = UpSampling1D(MP_ratio)(x)
x = Conv1D(channel_num1, filtersize, activation=HLactivation, padding='same', kernel_regularizer=regularizer)(x)
x = UpSampling1D(MP_ratio)(x)
decoded = Conv1D(1, filtersize, activation=outputactivation, padding='same')(x)

# Define the autoencoder model
autoencoder = Model(input_data, decoded)

# Compile the model using the weighted loss function
autoencoder.compile(optimizer='adam', loss=weighted_mse)

# Print a summary of the model to verify the architecture
autoencoder.summary()
