# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from funcs_dsp import *
from funcs_df import load_df

# Load dataframe and grab waveforms
laptop = True
dfs_to_load = ["Lots of Data", "York Data 2", "UWO Data"] # If this is empty, all are loaded
# dfs_to_load = ["Curated Data"]
df = load_df(laptop=laptop, dfs_to_load=dfs_to_load)
# Crop to only wf
df = df[df['wf'].notna()]
# Crop to only human/lizards
df_human = df[df['species'].isin(["Human"])]
df_lizard = df[df['species'].isin(["Lizard", "Anolis"])]
# Shuffle dfs
df_human = df_human.sample(frac=1).reset_index(drop=True)
df_lizard = df_lizard.sample(frac=1).reset_index(drop=True)

# Parameters
scaling = "density"
detrend = None # No detrending
win_type = 'boxcar'
nperseg = 32768
zpad = 1 # No zero padding
nfft = nperseg*zpad
log = True
f_min = 500 # Set minimum frequency


for i in range(50, 100):
    # Grab waveforms
    wf_human = df_human.iloc[i]['wf']
    wf_lizard = df_lizard.iloc[i]['wf']
    sr_human = df_human.iloc[i]['sr']
    sr_lizard = df_lizard.iloc[i]['sr']
    if sr_human == sr_lizard:
        fs = sr_human
    else:
        raise("Shouldn't these all have the same samplerate?")

    f_human, psd_human = welch(wf_human, scaling=scaling, fs=fs, window=win_type, nperseg=nperseg, nfft=nfft, detrend=detrend)
    f_lizard, psd_lizard = welch(wf_lizard, scaling=scaling, fs=fs, window=win_type, nperseg=nperseg, nfft=nfft, detrend=detrend)

    if np.array_equal(f_human, f_lizard):
        f = f_human
    else:
        raise("Why aren't these the same?")

    # Crop PSD and freq axis
    f_min_idx = np.argmin(np.abs(f - f_min)) # Convert frequency to index
    f_max_idx = f_min_idx + 4096

    for psd, species in zip([psd_human, psd_lizard], ["Human", "Lizard"]):
        plt.figure(figsize=(10, 5))
        if scaling == 'spectrum':
            label = "Power Spectrum"
            ylabel = "PS"
        elif scaling == 'density':
            label = "Power Spectral Density"
            ylabel = "PSD"
            
        if log:
            # Convert to log
            psd = 10 * np.log10(psd)
            ylabel += " (Log)"
        else:
            ylabel += " (Linear)"
        
        if detrend == "constant":
            label += " (Detrended)"
        elif detrend == False:
            label += " (Not Detrended)"
            
        label+=f": nperseg={nperseg}"
        plt.title(f"{species} PSD")
        plt.plot(f, psd, label="Full Spectrum")
        plt.plot(f[f_min_idx:f_max_idx], psd[f_min_idx:f_max_idx], label="Proposed Crop")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(f"{species} {i}.png")
        plt.close()

