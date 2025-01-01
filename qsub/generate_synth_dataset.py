import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import os
import math

from funcs_synth import *


# LOAD DATA
# First navigate to our directory
directory_path = os.path.join("Data", "spectra_and_noise_floors.parquet")
# Load the dataframe
df = pd.read_parquet(directory_path)
rng = np.random.default_rng()
f = rfftfreq(32768, 1/44100)[0:8192]
noise_floors = df['noise floor'].to_list()
spectra = df['spectrum'].to_list()
species_list = df['species'].to_list()
filepaths = df['filepath'].to_list()



# Make the transfer dataset
print("Synthesizing Transfer Dataset")
rows = []
i=0
for spectrum, noise_floor, species, filepath in zip(spectra, noise_floors, species_list, filepaths):
    i += 1
    print(f"Synthesizing {i} / {len(spectra)}")
    d = synthesize_spectrum(spectrum, noise_floor, f=f, species=species, filepath=filepath, noise_domain='linear')
    if any(math.isnan(x) for x in d['synth spectrum']):
        print("Excluding this one")
    else:
        rows.append(d)
synth_transfer_df = pd.DataFrame(rows)
synth_transfer_df.to_parquet("synth_transfer_df.parquet", engine='pyarrow')

# Make the general dataset
print("Synthesizing General Dataset")
rows = []
i = 0
for spectrum, noise_floor, species, filepath in zip(spectra, noise_floors, species_list, filepaths):
    i += 1
    print(f"Synthesizing {i} / {len(spectra)}")
    d = synthesize_spectrum(spectrum, noise_floor, f=f, species='General', filepath=filepath, noise_domain='linear')
    if any(math.isnan(x) for x in d['synth spectrum']):
        print("Excluding this one")
    else:
        rows.append(d)
synth_general_df = pd.DataFrame(rows)
synth_general_df.to_parquet("synth_general_df.parquet", engine='pyarrow')