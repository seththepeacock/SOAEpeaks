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
        path = r"Data/"
        
    if dfs_to_load == ["All"]:
        dfs_to_load = ["Pre-2014 Data", "Curated Data", "Extra Owl", "Lots of Data", "UWO Data", "York Data 1", "York Data 2", "York Data 3"]

    dfs = []
    for file in dfs_to_load:
        print(f"Loading {file}")
        df0 = pd.read_parquet(f"{path + file}.parquet")
        dfs.append(df0)
        
    print("Combining into one Dataframe!")
    return pd.concat(dfs, ignore_index=True)

def process_wfs(df):
    """Take FFT of all raw waveforms in the dataframe and convert to dB SPL"""
    # Define constants that allow us to correct for experimental setup
    amp_factor = 0.01  # Correct for amplifier gain
    # amp_factor = 0.1  # Correct for amplifier gain
    mic_factor = 10**(-6)  # Corresponds to the mic level of 1 microvolt
    # mic_factor = 1  # Corresponds to the mic level of 1 microvolt
    rms_factor = np.sqrt(2)  # Converts to RMS value

    # Define window length
    n_win = 32768

    # Track where we're at
    n_wf = (df['sr'] != 0).sum()
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
            np.abs(rfft(wf[i * n_win:(i + 1) * n_win])*(2/n_win))
            for i in range(len(wf) // n_win)
        ]

        # Average over all windows
        avg_mags = np.mean(mags_list, axis=0)
        db_SPL = 20 * np.log10(avg_mags * amp_factor * rms_factor / mic_factor)
        # Update the DataFrame directly
        df.at[idx, 'spectrum'] = db_SPL
        # Remove waveform for space
        df.at[idx, 'wf'] = []
        # No need to store freq ax, as these can be generated via rfftfreq(n_win, 1 / sr)
        
    return df

            
def homogenize_df(df, species=["Human", "Lizard", "Anolis"]):
    """ Throws out bad samples and crops longer spectra to the right length
      ----------------
      df: pandas dataframe
        Should already have processed waveforms
      species: list of strings
        List of species to keep
    """
    # just keep the species we want
    df = df[df['species'].isin(species)]
    
    # # crop any spectra that are longer than the minimum length (which will be more than enough)
    # # get minimum length (out of rows that actually have a spectrum)
    # min_num_bins = df[df['sr']==0]['spectrum'].apply(len).min()
    # for idx, row in df.iterrows():
    #     if len(row['spectrum']) > min_num_bins:
    #         df.at[idx, 'spectrum'] = row['spectrum'][0:min_num_bins]
    #         # if this is preprocessed spectrum, crop the frequency axis as well
    #         if row['sr']==0:
    #             df.at[idx, 'freqs'] = row['freqs'][0:min_num_bins]

    # crop spectra to length 8192
    cropped_length = 8192
    for idx, row in df.iterrows():
        df.at[idx, 'spectrum'] = row['spectrum'][0:cropped_length]
        # if this is preprocessed spectrum, crop the frequency axis as well
        if row['sr']==0:
            df.at[idx, 'freqs'] = row['freqs'][0:cropped_length]

    # Throw out the samples that don't have the right frequency axis bin width (there's just a few)
    # Set the desired bin width (almost all match this value)
    target_value = 1.3458
    # Function to compute the difference and round it
    def diff_is_target(row):
        # if it's a waveform, definitely keep
        if row['sr'] != 0:
            keep = True
        # Otherwise, Get the array from the 'freqs' column
        else:
            freqs = np.array(row['freqs'])
            diff = freqs[1] - freqs[0]
            keep = round(diff, 4) == round(target_value, 4)
        return keep

    # Filter the dataframe
    df = df[df.apply(diff_is_target, axis=1)]
    
    return df
            

def max_min_df(df):
    # Extract the spectra as a NumPy array for efficient computation
    spectra = np.array(df['spectrum'].to_list())
    
    # Compute the min and max for each spectrum
    spec_min = np.min(spectra, axis=1)
    spec_max = np.max(spectra, axis=1)
    
    # Perform max-min scaling
    scaled_spectra = (spectra - spec_min[:, None]) / (spec_max - spec_min)[:, None]
    
    # Add the scaled spectra and min/max values to the DataFrame
    df['scaled spectrum'] = list(scaled_spectra)
    df['max'] = spec_max
    df['min'] = spec_min
    
    return df

        
        