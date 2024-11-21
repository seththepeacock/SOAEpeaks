import numpy as np
from plots import *
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.fft import rfft, rfftfreq

def load_df(laptop=False, dfs_to_load=["York Data 1"]):
    """ Load all desired dataframes and concatenate (with no arguments, this loads just York Data 1 with OSC path)
    """
    if laptop:
        path = r"C:\Users\Owner\OneDrive\Documents\GitHub\SOAEpeaks\Data\\"
    else:
        path = ""
        
    if dfs_to_load == "All":
        dfs_to_load = ["Pre-2014 Data", "Curated Data", "Extra Owl", "Lots of Data", "UWO Data", "York Data 1", "York Data 2", "York Data 3"]

    dfs = []
    for file in dfs_to_load:
        print(f"Loading {file}")
        df0 = pd.read_parquet(f"{path + file}.parquet")
        dfs.append(df0)
        
    print("Combining into one Dataframe!")
    return pd.concat(dfs)

def process_wfs(df):
    """Take FFT of all waveforms in the dataframe and convert to dB SPL"""
    # Define constants that allow us to correct for experimental setup
    amp_factor = 0.01  # Correct for amplifier gain
    mic_factor = 10**(-6)  # Corresponds to the mic level of 1 microvolt
    rms_factor = np.sqrt(2)  # Converts to RMS value

    # Define window length
    n_win = 32768

    # Track where we're at
    n_wf = (df['sr'] == 0).sum()
    n_current = 0

    for idx, row in df.iterrows():
        # Skip pre-processed samples
        if row['sr'] == 0:
            continue

        n_current+=1
        print(f"Processing wf {n_current}/{n_wf}")

        # Get waveform and samplerate
        wf = row['wf']
        sr = row['sr']

        # Divide the waveform into windows of length n_win, take magnitude of FFT of each
        mags_list = [
            np.abs(rfft(wf[i * n_win:(i + 1) * n_win]))
            for i in range(len(wf) // n_win)
        ]

        # Average over all windows
        avg_mags = np.mean(mags_list, axis=0)
        db_SPL = 20 * np.log10(avg_mags * amp_factor * rms_factor / mic_factor)
        freqs = rfftfreq(n_win, 1 / sr)

        # Update the DataFrame directly
        df.at[idx, 'spectrum'] = db_SPL
        df.at[idx, 'freqs'] = freqs

    return df

            
def get_samples(processed_df, species=["Human", "Lizard", "Anolis"]):
    """ Throws out bad samples and max-min scales remaining; returns dict with list of samples, max, min, and (one) freq ax
      Parameters
      ----------------
      processed_df: pandas dataframe
        Meant to already have processed waveforms
      species: list of strings
        List of species to keep (exports each species as a separate file)
    """
    df = processed_df[processed_df['species'].isin(species)]
        
        