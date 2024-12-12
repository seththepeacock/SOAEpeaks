import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Concatenate, TimeDistributed, LSTM, Bidirectional, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.autograph.experimental import do_not_convert
from sklearn.model_selection import train_test_split
import numpy as np
from helper_funcs import gen_samples
from scipy.fft import rfftfreq
import pickle
import matplotlib.pyplot as plt
import time

# Load and split dataset

# File paths
transfer_directory_path = os.path.join("Data", "synth_transfer_df.parquet")
general_directory_path = os.path.join("Data", "synth_general_df.parquet")

# Load the dataframes
synth_transfer_df = pd.read_parquet(transfer_directory_path)
synth_general_df = pd.read_parquet(general_directory_path)

# Ensure column names match
assert list(synth_transfer_df.columns) == list(synth_general_df.columns), "Column names do not match!"

# Add a dataset identifier for stratification
synth_transfer_df['dataset'] = 'transfer'
synth_general_df['dataset'] = 'general'

# Combine the datasets
df = pd.concat([synth_transfer_df, synth_general_df], axis=0)
df.reset_index(drop=True, inplace=True)

# Random state for reproducibility
rs = 1

print(len(df))

# Split each dataset (transfer and general) independently
transfer_train, transfer_temp = train_test_split(
    synth_transfer_df,
    test_size=0.3,  # 30% of transfer dataset
    stratify=synth_transfer_df['species'],  # Stratify based on species
    random_state=rs
)

general_train, general_temp = train_test_split(
    synth_general_df,
    test_size=0.3,  # 30% of general dataset
    stratify=synth_general_df['species'],  # Stratify based on species
    random_state=rs
)

# Combine the training datasets from transfer and general
train_df = pd.concat([transfer_train, general_train], axis=0).reset_index(drop=True)

# Split temp datasets (transfer and general) into test and validation
transfer_test, transfer_val = train_test_split(
    transfer_temp,
    test_size=0.5,  # Split evenly into test and validation
    stratify=transfer_temp['species'],
    random_state=rs + 1
)

general_test, general_val = train_test_split(
    general_temp,
    test_size=0.5,  # Split evenly into test and validation
    stratify=general_temp['species'],
    random_state=rs + 1
)

# Combine test and validation datasets from transfer and general
test_df = pd.concat([transfer_test, general_test], axis=0).reset_index(drop=True)
val_df = pd.concat([transfer_val, general_val], axis=0).reset_index(drop=True)

# Prepare samples
print("Generating Training Samples")
X_train, y_train, mins_maxes_train, isolated_peaks_train = gen_samples(train_df)
print("Generating Test Samples")
X_test, y_test, mins_maxes_test, isolated_peaks_test = gen_samples(test_df)
print("Generating Validation Samples")
X_val, y_val, mins_maxes_val, isolated_peaks_val = gen_samples(val_df)

# Expand axes for conv layers
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


# Define custom loss and metric callbacks
def custom_loss(y_true, y_pred):
    """
    Custom loss function using predictions and weights.
    """
    # Compute binary cross-entropy loss
    bce_loss = tf.keras.backend.binary_crossentropy(y_true[:, :, 0], y_pred[:, :, 0])  # Shape (batch_size, N)

    # prom = y_true[:, :, 2] is the 3rd node label, shape (batch_size, N)
    # If prom < 0, weight is 1 (weight of BCE loss for non-peak bins), else apply weight_func
    weights = tf.where(y_true[:, :, 2] < 0, tf.ones_like(y_true[:, :, 2]), peak_encourage/(1+tf.exp(k-y_true[:, :, 2])))  
    # Apply weights
    weighted_bce_loss = bce_loss * weights  # Shape (batch_size, bins_per_sample)

    # Average loss across all samples and bins
    total_loss = tf.reduce_mean(weighted_bce_loss)  # Scalar

    return total_loss


def peak_counting_error(isolated_peaks, predictions, threshold_list, verbose=False):
    # Just grab the labels since we're ignoring width and height
    predictions = predictions[:, :, 0]
    M = isolated_peaks.shape[0] # Number of samples
    assert M == predictions.shape[0], "Mismatch in number of samples!"
    best_error = 10000  # Initial large value for minimizing error
    best_thresh = None
    for thresh in threshold_list:
        current_error = 0
        val_predictions_snapped = (predictions > thresh).astype(int)
        
        for row in range(M):
            predictions_row = val_predictions_snapped[row, :]
            isolated_labels_row = isolated_peaks[row, :]
            
            # We want to go along the predictions row, and find continuous chunks of 1s and 0s.
            # Every time a chunk ends, we then check how many peaks were truly in that chunk (by counting the 1s in those indices in isolated_labels_row)
            # We then add the square of the differences between the predicted number of peaks and the actual number of peaks to total_error
            # The predicted number of peaks in a chunk of 1s is always 1, and the predicted number of peaks in a chunk of 0s is always 0.
            
            # Track the current chunk
            current_chunk_value = predictions_row[0]
            current_chunk_start = 0

            for idx in range(1, len(predictions_row) + 1):  # +1 to handle the last chunk
                if idx == len(predictions_row) or predictions_row[idx] != current_chunk_value:
                    # Chunk ends here
                    chunk_end = idx
                    chunk_labels = isolated_labels_row[current_chunk_start:chunk_end]
                    
                    # Predicted peaks for this chunk
                    predicted_peaks = current_chunk_value
                    # Actual peaks for this chunk
                    actual_peaks = int(chunk_labels.sum())  # Count the 1s in the chunk
                    
                    # Add squared error to total_error
                    current_error += (predicted_peaks - actual_peaks) ** 2
                    
                    # Start a new chunk
                    current_chunk_value = predictions_row[idx] if idx < len(predictions_row) else None
                    current_chunk_start = idx
        
        current_error = current_error / M
        if verbose:
            print(f"Peak counting error for threshold {thresh}: {current_error}")
        # Update the best threshold if this one performs better
        if current_error < best_error:
            best_error = current_error
            best_thresh = thresh
    if verbose:
        print(f"Best threshold: {best_thresh}, Best Peak Counting Error: {best_error}")
    return best_error, best_thresh


# Calculate Peak Counting Error for all hyperparameter combos
epochs = 25
min_epoch = 1
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
lr = 0.001

for include_LSTM in [False, True]:
    for k in [0, 3]:
        for peak_encourage in [15, 5, 10, 1]:
            start_time = time.time()
            all_epoch_best_thresh = 0
            all_epoch_best_error = np.inf
            best_epoch = 0

            # Preload all models for the current hyperparameter combination
            models = {
                i: load_model(
                    os.path.join("PP Models (All Epochs)", f"V1_k-{k}_PE-{peak_encourage}_LSTM-{include_LSTM}_Epochs-{epochs}_LR-{lr} - epoch {i:02}.keras"),
                    custom_objects={"custom_loss": custom_loss}
                )
                for i in range(min_epoch, epochs + 1)
            }

            for i in range(min_epoch, epochs + 1):  # Use epochs instead of hardcoding 25
                model_version = f"V1_k-{k}_PE-{peak_encourage}_LSTM-{include_LSTM}_Epochs-{epochs}_LR-{lr} - epoch {i:02}"
                model = models[i]  # Use preloaded model
                val_pred = model.predict(X_val, verbose=0)  # Predicted probabilities
                best_error, best_thresh = peak_counting_error(isolated_peaks_val, val_pred, threshold_list=threshold_list)

                if best_error < all_epoch_best_error:
                    all_epoch_best_error = best_error
                    all_epoch_best_thresh = best_thresh
                    best_epoch = i

            # Load the best model using the preloaded dictionary
            best_model = models[best_epoch]
            test_pred = best_model.predict(X_test, verbose=0)  # Predicted probabilities
            test_error, test_thresh = peak_counting_error(isolated_peaks_test, test_pred, threshold_list=[all_epoch_best_thresh])
            assert test_thresh == all_epoch_best_thresh
            print(f"LSTM = {include_LSTM}, k = {k}, PE = {peak_encourage}, Best Val Threshold: {round(all_epoch_best_thresh, 2)}, Best Epoch: {best_epoch}, Test Error: {round(test_error, 5)}")
            elapsed_time = time.time() - start_time
            print(f"Execution time per model: {elapsed_time:.2f} seconds")
        


