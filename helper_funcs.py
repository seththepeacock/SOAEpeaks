import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.fft import rfft, rfftfreq
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter


def lorentzian(f, f0, hwhm, snr):
    return snr / (1 + ((f - f0) / hwhm)**2)

def gen_samples(df, hwhm_max, snr_max):
    # Allocate memory for data
    samples = np.zeros((len(df), 8192, 4))
    # Set the widths and snrs to None
    samples[:, :, 2] = np.nan
    samples[:, :, 3] = np.nan
    # Get synthetic spectra (rows, N bins)
    samples[:, :, 0] = np.stack(df['synthetic spectrum'])
    # Get labels (rows, num_peaks)
    f0s = np.array(df['f0'], dtype=object)
    hwhms = np.array(df['hwhm'], dtype=object)
    snrs = np.array(df['snr'], dtype=object)
    # Rescale peaks to be in range [0, 1]
    hwhms = hwhms / hwhm_max
    snrs = snrs / snr_max
    # Grab frequency array
    f = rfftfreq(32768, 1/44100)[0:8192]
    for row in range(len(df)):
        # Get the peak list components for this row
        f0s_i = f0s[row]
        hwhms_i = hwhms[row]
        snrs_i = snrs[row]
        if len(f0s_i) != len(hwhms_i) or len(f0s_i) != len(snrs_i):
            raise ValueError("f0s, hwhms, and snrs must have the same length!")
        # Add peak lists
        for peak_num in range(len(f0s_i)):
            # Get peak components
            f0 = f0s_i[peak_num]
            hwhm = hwhms_i[peak_num]
            snr = snrs_i[peak_num]
            # Get frequency array index corresponding to the peak location
            f0_index = np.where(f == f0)[0]
            # Add labels: [Yes/no peak, hwhm, snr]
            samples[row, f0_index, 1:] = [1, hwhm, snr]
    return samples
        
        
    
    

def gen_spectrum(f=rfftfreq(32768, 1/44100), species='Lizard', noise_floor=[], noise_sigma=0.01, f0_dist='chi', plot=False):
    # Get frequency axis
    f = rfftfreq(32768, 1/44100)
    
    # Get num bins in the spectrum
    n_bins = len(noise_floor)
    if n_bins != 8192:
        raise ValueError(f"Expected noise floor to be 8192 bins, but it's {n_bins} bins!")
    
    # Set generation parameters
    
    # f0: For the transfer dataset, we'll draw half from a chi square
    f0_chi_dof = 10
    f0_chi_og_pivot = 20 # This is the value of the original distribution that we want to "grab"
    f0_chi_new_pivot = 8000 # We'll rescale ("pull") the distribution so that that value becomes this value
    # and the other half from a uniform distribution
    f0_min = 50
    f0_max = 6000
    
    # hwhm: we'll draw from uniform distribution
    if species == 'Lizard':
        hwhm_min = 25
        hwhm_max = 150
    elif species == 'Human':
        hwhm_min = 1
        hwhm_max = 25
    elif species == 'General':
        hwhm_min = 1
        hwhm_max = 150

    # SNR: we'll draw from a uniform distribution
    snr_min = 0
    snr_max = 20
    
    # Number of peaks: we'll draw from a uniform distribution
    if species == 'Lizard':
        num_peaks_min = 5
        num_peaks_max = 20
    elif species == 'Human':
        num_peaks_min = 5
        num_peaks_max = 20
    elif species == 'General':
        num_peaks_min = 20
        num_peaks_mx = 30
    
    # Create a Generator instance
    rng = np.random.default_rng()  # Default random generator instance
    
    # Initialize matrix to store all the different peaks to be added to the noisefloor
    spec_components = np.empty((n_peaks + 2, n_bins))
    
    # Add noise floor
    spec_components[-1, :] = noise_floor
    
    # Add gaussian noise
    spec_components[-2, :] = rng.normal(0, noise_sigma, n_bins)
    
    # Pick number of peaks
    n_peaks = rng.integers(num_peaks_min, num_peaks_max, endpoint=True)
    
    # Store peak list
    peak_list = np.empty((n_peaks, 3))
    
    # Generate and add peaks
    for i in range(n_peaks):
        # f0
        if f0_dist == 'chi':
            # Draw the positions from a chi square
            f0 = rng.chisquare(df=f0_chi_dof)*f0_chi_new_pivot/f0_chi_og_pivot
            # make sure it's below our spectral max, resample if not
            while f0 > f[-1]:
                f0 = rng.chisquare(df=f0_chi_dof)*f0_chi_new_pivot/f0_chi_og_pivot
        elif f0_dist == 'uniform':
            # Draw the positions from a uniform distribution across the frequency range
            f0 = rng.uniform(f[0], f[-1])
        # Lock this onto the grid
        f0 = f[np.argmin(np.abs(f - f0))]
        
        # hwhm
        hwhm = rng.uniform(hwhm_min, hwhm_max)
        
        # SNR
        snr = rng.uniform(snr_min, snr_max)
        
        # Add peak
        spec_components[i, :] = lorentzian(f, f0, hwhm, snr)
        
        # Store peak info
        peak_list[i, :] = [f0, hwhm, snr]
        
    # Combine everything into the final synthetic spectrum
    synth_spectrum = np.sum(spec_components, axis=0)

    if plot:
        plt.plot(f/1000, synth_spectrum)
        plt.plot(f/1000, noise_floor, label="Noise Floor")
        plt.title(f"Synthetic Spectrum: Species={species}, {n_peaks} peaks, {f0_dist} location dist")
        plt.legend()
        plt.xlabel("Frequency (kHz)")
        plt.ylabel("dB SPL")
        plt.show()
    
    return {'noise floor' : noise_floor, 'species' : species, 'synth spectrum' : synth_spectrum, 'f0' : peak_list[:, 0], 'hwhm' : peak_list[:, 1], 'snr' : peak_list[:, 2]}

def get_noise_floor(f, spectrum, lowess_frac = 0.3, smoothing_sigma = 150, initial_cutoff = 300):
    lowess_fit = sm.nonparametric.lowess(spectrum, f, frac=lowess_frac, return_sorted=False)
    min_spectrum = np.minimum(lowess_fit, spectrum)
    min_spectrum[0:initial_cutoff] = spectrum[0:initial_cutoff]
    noise_floor = gaussian_filter(min_spectrum, sigma=smoothing_sigma)
    return noise_floor

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

        
        