import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import pandas as pd
import os
from helper_funcs import *

df_path = os.path.join("Data", "processed_df.parquet")
# Load the dataframe
df = pd.read_parquet(df_path)
spectrums = df['spectrum'].to_list()
filepaths = df['filepath'].to_list()
species = df['species'].to_list()
f = rfftfreq(32768, 1/44100)[0:8192]
rows = []

for i, (spectrum, filepath, species) in enumerate(zip(spectrums, filepaths, species)):
    noise_floor = get_noise_floor(f, spectrum)
    rows.append({"species":species, "noise floor":noise_floor, "filepath":filepath, "spectrum":spectrum})
    if i > 3:
        break

noise_floors = pd.DataFrame(rows)
noise_floors.to_parquet("spectra_and_noise_floors.parquet", engine='pyarrow')