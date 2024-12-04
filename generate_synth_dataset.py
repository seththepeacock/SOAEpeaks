import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import os

from helper_funcs import *


# LOAD DATA
# First navigate to our directory
directory_path = os.path.join("Data", "processed_df.parquet")
# Load the dataframe
df = pd.read_parquet(directory_path)
rng = np.random.default_rng()
f = rfftfreq(32768, 1/44100)[0:8192]
spectra = df['spectrum'].to_list()
species_list = df['species'].to_list()
filepaths = df['filepath'].to_list()    



# Make the transfer dataset
rows = []
i=0
for spectrum, species, filepath in zip(spectra, species_list, filepaths):
    i += 1
    print(f"Synthesizing {i} / {len(spectra)}")
    d = synthesize_spectrum(spectrum, f=f, species=species, filepath=filepath, noise_domain='log')
    rows.append(d)
synth_transfer_df = pd.DataFrame(rows)
synth_transfer_df.to_parquet("synth_transfer_df.parquet", engine='pyarrow')

# Make the general dataset
rows = []
i = 0
for spectrum, species, filepath in zip(spectra, species_list, filepaths):
    i += 1
    print(f"Synthesizing {i} / {len(spectra)}")
    d = synthesize_spectrum(spectrum, f=f, species='General', filepath=filepath, noise_domain='log')
    rows.append(d)
synth_general_df = pd.DataFrame(rows)
synth_general_df.to_parquet("synth_general_df.parquet", engine='pyarrow')