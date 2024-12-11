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

# Set parameters
k=3
peak_encourage=1
include_LSTM=False
epochs = 1
lr = 0.001

model_version = f"V1_k-{k}_PE-{peak_encourage}_LSTM-{include_LSTM}_Epochs-{epochs}_LR-{lr}"
batch_size = 32
patience = 3
threshold_list = [0.05, 0.1, 0.15, 0.2]

# First navigate to our directory
transfer_directory_path = os.path.join("Data", "synth_transfer_df.parquet")
general_directory_path = os.path.join("Data", "synth_general_df.parquet")
# Load the dataframes
synth_transfer_df = pd.read_parquet(transfer_directory_path)
synth_general_df = pd.read_parquet(general_directory_path)
# Concatenate (after making sure they share columns) and then reset indices
assert list(synth_transfer_df.columns) == list(synth_general_df.columns), "Column names do not match!"
df = pd.concat([synth_transfer_df, synth_general_df], axis=0)
df.reset_index(drop=True, inplace=True)

rs=1
# Split into train (70%) and temp (30%) with stratification
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df['species'],  # Stratify based on the 'species' column
    random_state=rs
)

# Split temp into test (15%) and validation (15%)
test_df, val_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['species'],  # Stratify again to maintain balance
    random_state=rs + 1
)

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



def peak_counting_error(isolated_peaks_val, val_predictions):
    # Just grab the labels since we're ignoring width and height
    val_predictions = val_predictions[:, :, 0]
    M = isolated_peaks_val.shape[0] # Number of samples
    assert M == val_predictions.shape[0], "Mismatch in number of samples!"
    best_error = 10000  # Initial large value for minimizing error
    best_thresh = None
    for thresh in threshold_list:
        current_error = 0
        val_predictions_snapped = (val_predictions > thresh).astype(int)
        
        for row in range(M):
            predictions_row = val_predictions_snapped[row, :]
            isolated_labels_row = isolated_peaks_val[row, :]
            
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
        print(f"Peak counting error for threshold {thresh}: {current_error}")
        # Update the best threshold if this one performs better
        if current_error < best_error:
            best_error = current_error
            best_thresh = thresh
    
    print(f"Best threshold: {best_thresh}, Best Peak Counting Error: {best_error}")
    return best_error, best_thresh
        

class ValidationMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, metric_name="peak_counting_error"):
        super(ValidationMetricCallback, self).__init__()
        self.validation_data = validation_data
        self.metric_name = metric_name
    
    @do_not_convert
    def on_epoch_end(self, epoch, logs=None):
        val_x, (val_y, isolated_peaks_val) = self.validation_data  # Unpack extra labels
        val_predictions = self.model.predict(val_x, verbose=0)
        val_predictions = val_predictions[:, :, 0]
        M = tf.shape(isolated_peaks_val)[0]  # Number of samples
        tf.assert_equal(M, tf.shape(val_predictions)[0], message="Mismatch in number of samples!")
        
        thresholds = tf.constant(threshold_list, dtype=tf.float32)
        best_error = tf.constant(1e10, dtype=tf.float32)  # Initial large value for minimizing error
        best_thresh = tf.constant(0.0, dtype=tf.float32)

        def calculate_error(thresh):
            val_predictions_snapped = tf.cast(val_predictions > thresh, tf.int32)

            def process_row(row_idx):
                predictions_row = val_predictions_snapped[row_idx]
                isolated_labels_row = isolated_peaks_val[row_idx]

                chunk_boundaries = tf.concat([[0], tf.where(tf.not_equal(predictions_row[:-1], predictions_row[1:]))[:, 0] + 1, [tf.size(predictions_row)]], axis=0)
                chunk_start_indices = chunk_boundaries[:-1]
                chunk_end_indices = chunk_boundaries[1:]

                def process_chunk(start, end):
                    chunk_labels = isolated_labels_row[start:end]
                    predicted_peaks = tf.cast(predictions_row[start], tf.float32)  # Cast to float32
                    actual_peaks = tf.cast(tf.reduce_sum(chunk_labels), tf.float32)  # Cast to float32
                    return tf.square(predicted_peaks - actual_peaks)

                squared_errors = tf.map_fn(
                    lambda indices: process_chunk(indices[0], indices[1]),
                    (chunk_start_indices, chunk_end_indices),
                    fn_output_signature=tf.float32
                )
                return tf.reduce_sum(squared_errors)

            total_error = tf.map_fn(process_row, tf.range(M), fn_output_signature=tf.float32)
            return tf.reduce_mean(total_error)

        def update_best(thresh, current_error, best_error, best_thresh):
            better = current_error < best_error
            return tf.cond(
                better,
                lambda: (current_error, thresh),
                lambda: (best_error, best_thresh)
            )

        for thresh in thresholds:
            current_error = calculate_error(thresh)
            best_error, best_thresh = update_best(thresh, current_error, best_error, best_thresh)

        # Round the best_thresh to 3 decimal places
        rounded_best_thresh = tf.round(best_thresh * 1000) / 1000.0  # Rounds to 3 decimal places
    
        # Add space before printing the custom output
        tf.print("\n--------------------------------------------")
        tf.print(f"Best threshold: {rounded_best_thresh}, Best Peak Counting Error: {best_error}")
        tf.print("--------------------------------------------\n")

        logs[self.metric_name] = best_error


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []  # List to store time per epoch

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()  # Record start time of epoch

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time  # Calculate epoch duration
        self.epoch_times.append(epoch_time)  # Save to list

# Define the input length / number of frequency bins (N)
N = 8192

# Input layer
input_layer = Input(shape=(N, 1), name="Input")
# Inception-like layer with 1D convolutions
convs = []
# We'll base our kernel choices on the hwhm distribution of the peaks. 
# Thin peaks are in 3Hz-10Hz range --> 5-15 bins
# Wide peaks are in 10Hz-100Hz range --> 15-149 bins
# We choose filters at a range of scales, odd (to facilitate being cenetered around a peak)
# and we want more filters for the medium-small range since there are more peaks at this scale.
# Otherwise largely arbitrarily.
kernels = [(3, 4), (5, 8), (9, 16), (15, 32), (31, 32), (55, 32), (71, 16), (101, 8), (149, 4), (201, 2)]
for kernel_size, num_filters in kernels:
    convs.append(Conv1D(num_filters, kernel_size=kernel_size, activation='relu', padding='same', name=f"Conv_{kernel_size}")(input_layer))

# Concatenate the outputs of all convolutional layers
concat_layer = Concatenate(name="Inception_Concat")(convs)

# Time Distributed Dense Layers
td_dense64 = TimeDistributed(Dense(64, activation='relu'), name="Dense_64")(concat_layer)
td_dense32A = TimeDistributed(Dense(32, activation='relu'), name="Dense_32A")(td_dense64)
if include_LSTM:
    bd_LSTM = Bidirectional(LSTM(16, return_sequences=True), name="LSTM")(td_dense32A)
    td_dense32B = TimeDistributed(Dense(32, activation='relu'), name="Dense_32B")(bd_LSTM)
else:
    td_dense32B = TimeDistributed(Dense(32, activation='relu'), name="Dense_32B")(td_dense32A)
td_dense16 = TimeDistributed(Dense(16, activation='relu'), name="Dense_16")(td_dense32B)

# Final layer with 3 outputs per input bin
output_layer = TimeDistributed(Dense(3, activation='sigmoid'), name="Output")(td_dense16)

# Define the model to output both predictions and weights
model = tf.keras.Model(
    inputs=input_layer, 
    outputs=output_layer,  # Explicitly define both outputs
    name=model_version
)


# Compile the model (lambda function in loss to allow for prominences to be passed in as weights)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    # loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, weights)
    loss=custom_loss
)

model.summary()


start_time = time.time()
model_path = os.path.join("PP Model", f"{model_version}.keras")

time_callback = TimeHistory()

# Add callbacks for better training
callbacks = [
    ValidationMetricCallback(validation_data=(X_val, (y_val, isolated_peaks_val)), metric_name="peak_counting_error"),
    EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),  # Stop if no improvement for 5 epochs
    ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),  # Save the best model
    time_callback
]

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

# Validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)


# Train the model
history = model.fit(
    dataset,
    validation_data=validation_dataset,
    epochs=epochs,        # Number of epochs
    batch_size=batch_size,  # Batch size
    callbacks=callbacks,    # Add callbacks for early stopping and checkpointing
    verbose=1               # Verbose output
)

history.history['epoch_times'] = time_callback.epoch_times

with open(os.path.join("PP Model", f"{model_version}_history.pkl"), 'wb') as file:
    pickle.dump(history.history, file)
    
elapsed_time = time.time() - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")