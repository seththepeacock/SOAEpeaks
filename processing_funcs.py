import numpy as np
from plots import *
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.fft import rfft, rfftfreq

def load_df(laptop=True, laptop_filenames=["UWO Data"]):
    """ Load all the separate dataframes and concatenate (with no arguments, this loads all the data with OSC paths)
    """
    if laptop:
        path = r"C:\Users\Owner\OneDrive\Documents\GitHub\SOAEpeaks\Data\\"
        filenames = laptop_filenames
    else:
        path = ""
        filenames = ["Pre-2014 Data", "Curated Data", "Extra Owl", "Lots of Data", "UWO Data", "York Data 1", "York Data 2", "York Data 3"]

    dfs = []
    for file in filenames:
        print(f"Loading {file}")
        df0 = pd.read_parquet(f"{path + file}.parquet")
        dfs.append(df0)
        
    print("Combining into one Dataframe!")
    return pd.concat(dfs)

def process_wfs(df):
    """ Take fft of all waveforms in the dataframe and convert to dB SPL
    """
    # Define constants that allow us to correct for experimental setup
    amp_factor = 0.01 # correct for amplifier gain
    mic_factor = 10**(-6) # corresponds to the mic level of 1 microvolt
    rms_factor = np.sqrt(2) # converts to RMS value
    
    for row in df.iterrows():
        # Skip pre-processed samples
        if row['sr'] == 0:
            continue
        # Get waveform and samplerate
        wf = row['wf']
        sr = row['sr']

        # divide up the waveform into windows of length n_win, take magnitude of fft of each
        n_win = 32768
        mags_list = []
        for i in range(int(len(wf)/n_win)):
            wf_win = wf[i*n_win:(i+1)*n_win]
            mags = np.abs(rfft(wf_win))
            mags_list.append(mags)
        # average over all windows
        avg_mags = np.mean(mags_list, axis=0)
        db_SPL = 20*np.log10(avg_mags*amp_factor*rms_factor/mic_factor)
        row['spectrum'] = db_SPL
        row['freqs'] = rfftfreq(n_win, 1/sr)
        
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
        
        