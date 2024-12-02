import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.fft import rfft, rfftfreq
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter


def lorentzian(f, f0, hwhm, snr):
    return snr / (1 + ((f - f0) / hwhm)**2)

def gen_samples(df, hwhm_max=100, snr_max=25):
    # Allocate memory for y (peak labels)
    y = np.zeros((len(df), 8192, 3))
    # Set the widths and snrs to -1
    y[:, :, 1] = -1
    y[:, :, 2] = -1
    # Get labels (rows, num_peaks)
    f0s = np.array(df['f0'], dtype=object)
    hwhms = np.array(df['hwhm'], dtype=object)
    snrs = np.array(df['snr'], dtype=object)
    # Rescale peaks to be in range ~[0, 1]
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
            y[row, f0_index, :] = [1, hwhm, snr]
    # Return X (rows, N bins) and y
    return np.stack(df['synthetic spectrum']), y

# def estimate_noise_sigma(sample_spectrum, f=rfftfreq(32768, 1/44100), noise_domain='log'):
#     sigmas = np.empty(16)
#     # First, we chop up into 16 chunks
#     for i in range(16):
#         f_win = f[i*500:(i+1)*500]
#         sample_spectrum_win = sample_spectrum[i*500:(i+1)*500]
#         # Next, we convert to linear scale (if specified)
#         if noise_domain == 'linear':
#             sample_spectrum_win = 10**(sample_spectrum_win/20)
#         # Now we fit a noise floor estimate using LOWESS
#         lowess = sm.nonparametric.lowess
#         # This controls how tightly the fit is to the data
#         frac = 0.01 # This value yielded a close but still smooth fit (too closely and we just follow the noise!)
#         noise_floor = lowess(sample_spectrum_win, f_win, frac=frac, return_sorted=False)
#         plt.plot(f_win, noise_floor, label='Noise Floor')
#         plt.plot(f_win, sample_spectrum_win, label='Sample Spectrum', alpha=0.5)
#         plt.show()
#         # The additive noise is then the difference between the sample spectrum and the noise floor
#         additive_noise = sample_spectrum_win - noise_floor
#         # We assume this noise is normally distributed with mean zero. 
#         # We now estimate the standard deviation using the MLE (since the mean is known, this is also unbiased):
#         sigmas[i] = np.sqrt(np.mean((additive_noise)**2))
#     print(sigmas)
#     return np.mean(sigmas)
        
        
# def estimate_and_add_noise(synth_spectrum, sample_spectrum, f=rfftfreq(32768, 1/44100), rng=np.random.default_rng(), noise_domain='log'):
#     # First we convert to linear scale (if specified)
#     if noise_domain == 'linear':
#         synth_spectrum = 10**(synth_spectrum/20)
#     # Now we estimate the standard deviation of the additive noise of this sample spectrum
#     sigma = estimate_noise_sigma(sample_spectrum, f=f, noise_domain=noise_domain)
#     # We can now add gaussian noise to the linear spectrum
#     noisy_synth_spectrum = synth_spectrum + rng.normal(loc=0, scale=sigma, size=len(synth_spectrum))
#     # Finally, we convert it back to dB (if necessary)
#     if noise_domain == 'linear':
#         noisy_synth_spectrum = 20 * np.log10(noisy_synth_spectrum)
#     # and return
#     return noisy_synth_spectrum
    
    
def estimate_noise_sigma(sample_spectrum, f=rfftfreq(32768, 1/44100), noise_domain='log'):
    # First, we crop frequency axis and sample spectrum to 8-11kHz as these won't have any peaks and are just noise
    f_min = np.argmin(np.abs(f - 8000))
    f_max = np.argmin(np.abs(f - 11000))
    f = f[f_min:f_max]
    sample_spectrum = sample_spectrum[f_min:f_max]
    # Next, we convert to linear scale (if specified)
    if noise_domain == 'linear':
        sample_spectrum = 10**(sample_spectrum/20)
    # Now we fit a noise floor estimate using LOWESS
    lowess = sm.nonparametric.lowess
    # This controls how tightly the fit is to the data
    frac = 0.1 # This value yielded a close but still smooth fit (too closely and we just follow the noise!)
    noise_floor = lowess(sample_spectrum, f, frac=frac, return_sorted=False)
    # The additive noise is then the difference between the sample spectrum and the noise floor
    additive_noise = sample_spectrum - noise_floor
    # We assume this noise is normally distributed with mean zero. 
    # We now estimate the standard deviation using the MLE (since the mean is known, this is also unbiased):
    sigma = np.sqrt(np.mean((additive_noise)**2))
    return sigma
        
        
def estimate_and_add_noise(synth_spectrum, sample_spectrum, f=rfftfreq(32768, 1/44100), rng=np.random.default_rng(), noise_domain='log'):
    # First we convert to linear scale (if specified)
    if noise_domain == 'linear':
        synth_spectrum = 10**(synth_spectrum/20)
    # Now we estimate the standard deviation of the additive noise of this sample spectrum
    sigma = estimate_noise_sigma(sample_spectrum, f=f, noise_domain=noise_domain)
    # We can now add gaussian noise to the linear spectrum
    noisy_synth_spectrum = synth_spectrum + rng.normal(loc=0, scale=sigma, size=len(synth_spectrum))
    # Finally, we convert it back to dB (if necessary)
    if noise_domain == 'linear':
        noisy_synth_spectrum = 20 * np.log10(noisy_synth_spectrum)
    # and return
    return noisy_synth_spectrum
   

def synthesize_spectrum(sample_spectrum, f=rfftfreq(32768, 1/44100), species='General', filepath=None, plot=False, noise_domain='log'):
    # Get noise floor
    noise_floor = get_noise_floor(f, sample_spectrum)
    
    # Get num bins
    n_bins = len(noise_floor)
    if n_bins != 8192:
        raise ValueError(f"Expected noise floor to be 8192 bins, but it's {n_bins} bins!")
    
    # Set generation parameters
    
    # f0: For the transfer dataset, we'll draw from a chi square (for general, just uniform across length of data)
    f0_chi_dof = 5
    f0_chi_og_pivot = 10 # This is the value of the original distribution that we want to "grab"
    f0_chi_new_pivot = 6000 # We'll rescale ("pull") the distribution so that that value becomes this value
    
    # This is the minimum distance between peaks (so as to not penalize model for missing right peaks directly on top of each other)
    f0_min_dist = 10

    # hwhm: we'll draw from uniform distributions; 90% from thin and 10% from wide for human, vice versa for lizard
    hwhm_min_thin = 1
    hwhm_max_thin = 10
    hwhm_min_wide = 10
    hwhm_max_wide = 100
    # For general, we'll draw half from thin and half from wide
    
    # SNR: we'll draw from a chi-square distribution
    snr_chi_dof = 7
    snr_chi_og_pivot = 10 # This is the value of the original distribution that we want to "grab"
    snr_chi_new_pivot = 10 # We'll rescale ("pull") the distribution so that that value becomes this value
    # snr_max = 30
    
    # Number of peaks: we'll draw from a uniform distribution
    if species in ['Lizard', 'Anolis']:
        num_peaks_min = 15
        num_peaks_max = 25
    elif species == 'Human':
        num_peaks_min = 15
        num_peaks_max = 25
    elif species == 'General':
        # We can fit more in since they're no longer concentrated with the chi-square
        num_peaks_min = 35
        num_peaks_max = 50
        
    
    # Create a Generator instance
    rng = np.random.default_rng()  # Default random generator instance
    
    # Pick number of peaks
    n_peaks = rng.integers(num_peaks_min, num_peaks_max, endpoint=True)
    
    # Initialize matrix to store all the different peaks to be added to the noisefloor
    spec_components = np.empty((n_peaks + 1, n_bins))
    
    # Add noise floor
    spec_components[-1, :] = noise_floor
    
    # Create empty peak list
    peak_list = np.empty((n_peaks, 3))
    
    # Generate and add peaks
    for i in range(n_peaks):
        # Peak Position (f0):
        # Keep picking until we get a new peak position
        f0 = None
        while f0 is None or np.abs(peak_list[:, 0] - f0).min() < f0_min_dist:
            # For transfer dataset, we draw from a chi-square
            if species in ["Lizard", "Human", "Anolis"]:
                # Draw positions from a chi-square and ensure it's below the spectral max
                f0 = None  # Initialize to ensure the loop starts
                while f0 is None or f0 > f[-1]:
                    f0 = rng.chisquare(df=f0_chi_dof) * f0_chi_new_pivot / f0_chi_og_pivot
            elif species == 'General':
                # Draw the positions from a uniform distribution across the frequency range
                f0 = rng.uniform(f[0], f[-1])
            # Lock this onto the grid
            f0 = f[np.argmin(np.abs(f - f0))]
        
        # Width (HWHM): 
        
        # General dataset is half thin and half wide
        if species == 'General':
            if rng.random() < 0.5:
                hwhm = rng.uniform(hwhm_min_thin, hwhm_max_thin)
            else:
                hwhm = rng.uniform(hwhm_min_wide, hwhm_max_wide)
        else:
            # For transfer dataset, we pick 90% of them from the corresponding uniform dist (lizard=wide, human=thin)
            if rng.random() < 0.90:
                if species == 'Human':
                    hwhm = rng.uniform(hwhm_min_thin, hwhm_max_thin)
                elif species in ['Lizard', 'Anolis']:
                    hwhm = rng.uniform(hwhm_min_wide, hwhm_max_wide)
            # Every 1/10 peak, we pick from the opposite to see how they mix
            else:
                if species == 'Human':
                    hwhm = rng.uniform(hwhm_min_wide, hwhm_max_wide)
                elif species in ['Lizard', 'Anolis']:
                    hwhm = rng.uniform(hwhm_min_thin, hwhm_max_thin)


        
        # SNR
        # Draw positions from a chi-square
        snr = rng.chisquare(df=snr_chi_dof) * snr_chi_new_pivot / snr_chi_og_pivot
        
        # Add peak
        spec_components[i, :] = lorentzian(f, f0, hwhm, snr)
        
        # Store peak info
        peak_list[i, :] = [f0, hwhm, snr]
        
    # Combine everything into the final synthetic spectrum
    synth_spectrum = np.sum(spec_components, axis=0)
    
    # Add gaussian noise
    synth_spectrum  = estimate_and_add_noise(synth_spectrum, sample_spectrum, f=f, rng=rng, noise_domain=noise_domain)
    
    if plot:
        if species == 'General':
            f0_dist = 'Uniform'
        elif species in ['Lizard', 'Human', 'Anolis']:
            f0_dist = 'Chi Square'

        # Compute the global y-limits for both graphs
        all_values = [synth_spectrum, noise_floor, sample_spectrum]
        global_min = min([min(data) for data in all_values])
        global_max = max([max(data) for data in all_values])

        # Create the subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot the synthetic spectrum on the first subplot
        axs[0].plot(f / 1000, synth_spectrum, label="Synthetic Spectrum")
        axs[0].plot(f / 1000, noise_floor, label="Noise Floor")
        axs[0].set_title(f"Synthetic Spectrum: species={species}, {n_peaks} peaks, {f0_dist} position distribution")
        axs[0].set_xlabel("Frequency (kHz)")
        axs[0].set_ylabel("dB SPL")
        axs[0].legend()
        axs[0].set_ylim(global_min, global_max)  # Set common y-limits
        peak_indices = np.argmin(np.abs(f[:, None] - peak_list[:, 0]), axis=0)
        axs[0].scatter(peak_list[:, 0] / 1000, synth_spectrum[peak_indices], color='red', label="Detected Peaks")
        # Plot the sample spectrum with noise floor on the second subplot
        axs[1].plot(f / 1000, sample_spectrum, label="Sample Spectrum")
        axs[1].plot(f / 1000, noise_floor, label="Noise Floor")
        axs[1].set_title(f"Sample Spectrum: {filepath if filepath else 'No filepath provided'}")
        axs[1].set_xlabel("Frequency (kHz)")
        axs[1].set_ylabel("dB SPL")
        axs[1].legend()
        axs[1].set_ylim(global_min, global_max)  # Set common y-limits

        # Adjust the layout and show the plots
        plt.tight_layout()
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
    mic_factor = 10**(-6)  # Corresponds to the mic level of 1 microvolt
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

        
        