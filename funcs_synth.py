import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import rfft, rfftfreq
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter
from scipy.signal import welch, get_window

def lorentzian(f, f0, hwhm, prominence):
    return prominence / (1 + ((f - f0) / hwhm)**2)

def get_label_hw(hwhm):
    return ((14 / 95) * hwhm + 5 / 19).astype(int)

def gen_samples(df, hwhm_max=100, prominence_max=25):
    f_length = 8192
    # Allocate memory for y (peak labels)
    y = np.zeros((len(df), f_length, 3))
    # And for y_isolated_peaks
    y_isolated_peaks = np.zeros((len(df), f_length))
    # Set the widths and prominences to -1
    y[:, :, 1] = -1
    y[:, :, 2] = -1
    # Get labels (rows, num_peaks)
    f0_i_s = np.array(df['f0_i'], dtype=object)
    hwhms = np.array(df['hwhm'], dtype=object)
    prominences = np.array(df['prominence'], dtype=object)
    # Rescale peaks to be in range ~[0, 1] (prominence could in principle be > 25 but this would be rare)
    hwhms_scaled = hwhms / hwhm_max
    prominences_scaled = prominences / prominence_max
    # Grab frequency array
    for row in range(len(df)):
        # Get the peak list components for this row
        f0_i_s_row = f0_i_s[row]
        hwhms_scaled_row = hwhms_scaled[row]
        hwhms_row = hwhms[row]
        prominences_scaled_row = prominences_scaled[row]
        if len(f0_i_s_row) != len(hwhms_scaled_row) or len(f0_i_s_row) != len(prominences_scaled_row):
            raise ValueError("f0s, hwhms, and prominences must have the same length!")
        # Add peak lists
        for peak_num in range(len(f0_i_s_row)):
            # Get peak components
            f0_i = int(f0_i_s_row[peak_num])
            hwhm_scaled = hwhms_scaled_row[peak_num]
            prominence_scaled = prominences_scaled_row[peak_num]
            # First add isolated position labels
            y_isolated_peaks[row, f0_i] = 1
            # Now we need to calculate how many bins +\- to label
            label_hw = get_label_hw(hwhms_row[peak_num])
            # min/max is to make sure we don't go out of bounds
            f0_i_labels = np.arange(max(f0_i - label_hw, 0), min(f0_i + label_hw + 1, f_length - 1))
            for bin_to_label in f0_i_labels:
                # Add labels: [Yes/no peak, hwhm_scaled, prominence]
                y[row, bin_to_label, :] = [1, hwhm_scaled, prominence_scaled]
    X_x = np.stack(df['synth spectrum'])
    min_vals = np.min(X_x, axis=1)  # Minimum along the rows
    max_vals = np.max(X_x, axis=1)  # Maximum along the rows
    mins_maxes = {"mins": min_vals, "maxes": max_vals}
    # Rescale
    X_x = (X_x - min_vals[:, None]) / (max_vals[:, None] - min_vals[:, None])
    # Return X (rows, N bins), y, and mins/maxes
    return X_x, y, mins_maxes, y_isolated_peaks 
    
def estimate_noise_sigma(parent_spectrum, f=rfftfreq(32768, 1/44100), noise_domain='linear'):
    # First, we crop frequency axis and sample spectrum to 8-11kHz as these won't have any peaks and are just noise
    f_min = np.argmin(np.abs(f - 8000))
    f_max = np.argmin(np.abs(f - 11000))
    f = f[f_min:f_max]
    parent_spectrum = parent_spectrum[f_min:f_max]
    # Next, we convert to linear scale (if specified)
    if noise_domain == 'linear':
        parent_spectrum = 10**(parent_spectrum/20)
    # Now we fit a noise floor estimate using LOWESS
    lowess = sm.nonparametric.lowess
    # This controls how tightly the fit is to the data
    frac = 0.1 # This value yielded a close but still smooth fit (too closely and we just follow the noise!)
    noise_floor = lowess(parent_spectrum, f, frac=frac, return_sorted=False)
    # The additive noise is then the difference between the sample spectrum and the noise floor
    additive_noise = parent_spectrum - noise_floor
    # We assume this noise is normally distributed with mean zero. 
    # We now estimate the standard deviation using the MLE (since the mean is known, this is also unbiased):
    sigma = np.sqrt(np.mean((additive_noise)**2))
    return sigma
        
def estimate_and_add_noise(synth_spectrum, parent_spectrum, f=rfftfreq(32768, 1/44100), rng=np.random.default_rng(), noise_domain='linear'):
    # First we convert to linear scale (if specified)
    if noise_domain == 'linear':
        synth_spectrum = 10**(synth_spectrum/20)
    # Now we estimate the standard deviation of the additive noise of this sample spectrum
    sigma = estimate_noise_sigma(parent_spectrum, f=f, noise_domain=noise_domain)
    # We can now add gaussian noise to the linear spectrum
    # Ensure no negative values after adding noise
    noisy_synth_spectrum = synth_spectrum + rng.normal(loc=0, scale=sigma, size=len(synth_spectrum))

    # Finally, we convert it back to dB (if necessary)
    if noise_domain == 'linear':
        return 20 * np.log10(noisy_synth_spectrum)
    else:
        return noisy_synth_spectrum

def synthesize_spectrum(parent_spectrum, noise_floor, f=rfftfreq(32768, 1/44100), species='General', filepath=None, plot=False, save_name=None,noise_domain='linear'):
    # Get num bins
    n_bins = len(noise_floor)
    if n_bins != 8192:
        raise ValueError(f"Expected noise floor to be 8192 bins, but it's {n_bins} bins!")
    
    # Rename Anolis --> Lizard
    if species == 'Anolis':
        species = 'Lizard'
        
    # Set generation parameters
    
    # f0: For the transfer dataset, we'll draw from a chi square (for general, just uniform across length of data)
    f0_chi_dof = 5
    f0_chi_og_pivot = 10 # This is the value of the original distribution that we want to "grab"
    f0_chi_new_pivot = 6000 # We'll rescale ("pull") the distribution so that that value becomes this value
    # For lizards, this chi square draw will set the center of a uniform distribution to draw from with width
    lizard_center_chi_dof = 23
    lizard_center_chi_og_pivot = 10
    lizard_center_chi_new_pivot = 1200
    lizard_unif_width = 1500
    
    # This is the minimum distance between peaks (so as to not penalize model for missing right peaks directly on top of each other)
    f0_min_dist = 50

    # hwhm: we'll draw from uniform distributions; 90% from thin and 10% from wide for human, vice versa for lizard
    hwhm_min_thin = 3
    hwhm_max_thin = 10
    hwhm_min_wide = 10
    hwhm_max_wide = 100
    # For general, we'll draw half from thin and half from wide
    
    # prominence: we'll draw from a chi-square distribution
    prominence_chi_dof = 7
    prominence_chi_og_pivot = 10 # This is the value of the original distribution that we want to "grab"
    prominence_chi_new_pivot = 8 # We'll rescale ("pull") the distribution so that that value becomes this value
    # slightly different one for humans (higher peaks)
    human_prominence_chi_dof = 7
    human_prominence_chi_og_pivot = 10 # This is the value of the original distribution that we want to "grab"
    human_prominence_chi_new_pivot = 10 # We'll rescale ("pull") the distribution so that that value becomes this value
    
    # Create a Generator instance
    rng = np.random.default_rng()  # Default random generator instance
    
    # Number of peaks: we'll draw from chi-squares for Human/Lizard, uniform for General    
    if species == 'General':
        gen_npeaks_min = 20
        gen_npeaks_max = 35
        npeaks = rng.integers(gen_npeaks_min, gen_npeaks_max, endpoint=True)
    elif species == 'Lizard':
        npeaks_chi_dof = 11
        npeaks_chi_og_pivot = 20 # This is the value of the original distribution that we want to "grab"
        npeaks_chi_new_pivot = 18 # We'll rescale ("pull") the distribution so that that value becomes this value
        npeaks = int(rng.chisquare(npeaks_chi_dof)/npeaks_chi_og_pivot*npeaks_chi_new_pivot)  
    elif species == 'Human':
        npeaks_chi_dof = 3
        npeaks_chi_og_pivot = 4 # This is the value of the original distribution that we want to "grab"
        npeaks_chi_new_pivot = 5 # We'll rescale ("pull") the distribution so that that value becomes this value
        npeaks = int(rng.chisquare(npeaks_chi_dof)/npeaks_chi_og_pivot*npeaks_chi_new_pivot)  
    else:
        raise ValueError("Unknown species!")

    
    # Initialize matrix to store all the different peaks to be added to the noisefloor
    spec_components = np.empty((npeaks + 1, n_bins))
    
    # Add noise floor
    spec_components[-1, :] = noise_floor
    
    # Create empty peak list
    peak_list = np.empty((npeaks, 3))
    # And a temporary one for spreading out peaks
    f0s = [-1000]
    
    if species == 'Lizard':
        # Pick f0 to be the center of the distribution and ensure it's below the spectral max
        lizard_center = None  # Initialize to ensure the loop starts
        while lizard_center is None or lizard_center > f[-1]:
            lizard_center = rng.chisquare(df=lizard_center_chi_dof) * lizard_center_chi_new_pivot / lizard_center_chi_og_pivot
    
    # Generate and add peaks
    for i in range(npeaks):
        # Peak Position (f0):
        if species == 'Lizard' and rng.uniform(0, 1) < 0.9:
            # For lizards, we grab 90% from a uniform distribution that is centered around the picked lizard_center
            f0 = None
            while f0 is None or np.abs(np.array(f0s) - f0).min() < f0_min_dist:
                f0 = rng.uniform(lizard_center - lizard_unif_width, lizard_center + lizard_unif_width)
        else:
            # This was the original plan; we just grab from the chi square distribution
            # Keep picking until we get a new peak position
            f0 = None
            while f0 is None or np.abs(np.array(f0s) - f0).min() < f0_min_dist:
                # For transfer dataset, we draw from a chi-square
                if species in ["Human", "Lizard"]:
                    # Draw positions from a chi-square and ensure it's below the spectral max
                    f0 = None  # Initialize to ensure the loop starts
                    while f0 is None or f0 > f[-1]:
                        f0 = rng.chisquare(df=f0_chi_dof) * f0_chi_new_pivot / f0_chi_og_pivot
                elif species == 'General':
                    # Draw the positions from a uniform distribution across the frequency range
                    f0 = rng.uniform(f[0], f[-1])
        # Now we have an f0; lock this onto the grid and add to f0s
        f0_i = np.argmin(np.abs(f - f0))
        f0 = f[f0_i]
        f0s.append(f0)
        
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


        
        # Draw prominence
        if species == 'Human':
            prominence = rng.chisquare(df=human_prominence_chi_dof) * human_prominence_chi_new_pivot / human_prominence_chi_og_pivot
        else:
            prominence = rng.chisquare(df=prominence_chi_dof) * prominence_chi_new_pivot / prominence_chi_og_pivot
           
        # Add peak
        spec_components[i, :] = lorentzian(f, f0, hwhm, prominence)
        
        # Store peak info
        peak_list[i, :] = [f0_i, hwhm, prominence]
        
        
    # Combine everything into the final synthetic spectrum
    synth_spectrum = np.sum(spec_components, axis=0)
    
    # Add gaussian noise
    synth_spectrum  = estimate_and_add_noise(synth_spectrum, parent_spectrum, f=f, rng=rng, noise_domain=noise_domain)
    
    
    if plot:
        if species == 'General':
            f0_dist = 'Uniform'
        elif species in ['Lizard', 'Human', 'Anolis']:
            f0_dist = 'Chi Square'

        # Compute the global y-limits for both graphs
        all_values = [synth_spectrum[100:], noise_floor[100:], parent_spectrum[100:]]
        global_min = min([min(data) for data in all_values])
        global_max = max([max(data) for data in all_values])

        # Create the subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot the synthetic spectrum on the first subplot
        axs[0].plot(f / 1000, synth_spectrum, label="Synthetic Spectrum", zorder=1)
        axs[0].plot(f / 1000, noise_floor, label="Noise Floor", color='orange', zorder=2)
        axs[0].set_title(f"Synthetic {species} Spectrum", fontsize=16)
        axs[0].set_xlabel("Frequency (kHz)", fontsize=14)
        axs[0].set_ylabel("dB SPL", fontsize=14)
        axs[0].set_ylim(global_min, global_max)  # Set common y-limits
        peak_indices = peak_list[:, 0].astype(int)
        axs[0].scatter(f[peak_indices] / 1000, synth_spectrum[peak_indices], color='red', label="True Peaks", s=20, zorder=3)
        axs[0].legend(fontsize=12)
        # Plot the sample spectrum with noise floor on the second subplot
        axs[1].plot(f / 1000, parent_spectrum, label="Sample Spectrum")
        axs[1].plot(f / 1000, noise_floor, label="Noise Floor", color='orange')
        # axs[1].set_title(f"Parent Spectrum: {filepath if filepath else 'No filepath provided'}")
        axs[1].set_title(f"Parent Spectrum", fontsize=16)
        axs[1].set_xlabel("Frequency (kHz)", fontsize=14)
        axs[1].set_ylabel("dB SPL", fontsize=14)
        axs[1].legend(fontsize=12)
        axs[1].set_ylim(global_min, global_max)  # Set common y-limits
        plt.tight_layout()

        # Save the figure or show plot
        if save_name:
            plt.savefig(save_name)
        else: 
            # Show the plots
            plt.show()

    
    return {'noise floor' : noise_floor, 'species' : species, 'synth spectrum' : synth_spectrum, 'f0_i' : peak_list[:, 0], 'hwhm' : peak_list[:, 1], 'prominence' : peak_list[:, 2]}

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

def lab_rescale(input):
    # Define constants that allow us to correct for lab setup
    amp_factor = 0.01  # Correct for amplifier gain
    mic_factor = 10**(-6)  # Corresponds to the mic level of 1 microvolt
    rms_factor = np.sqrt(2)  # Converts to RMS value
    
    # Rescale input (either wf or magnitudes, NOT powers!)
    input = input * amp_factor / (mic_factor * rms_factor)

def get_welch_og_chris(wf, rescale=True):
    # Define window length
    n_win = 32768
    
    # Divide the waveform into windows of length n_win, take magnitude of FFT of each
    mags_list = [
        np.abs(rfft(wf[i * n_win:(i + 1) * n_win])*(2/n_win))
        for i in range(len(wf) // n_win)
    ]

    # Average over all windows
    avg_mags = np.mean(mags_list, axis=0)
    
    # Rescale
    if rescale:
        avg_mags = lab_rescale(avg_mags)
        
    return 20 * np.log10(avg_mags)
    
def get_welch(wf, shift_prop=0.5, win_type='hann', n_win=32768, num_wins=None, dB=True, rescale=True):
    # Define window length
    n_shift = int(n_win*shift_prop)
    
    # Rescale waveform
    if rescale:
        wf = lab_rescale(wf)
    
    n_possible_win_starts = len(wf) - n_win # Number of samples that could be a start of a sample
    n_full_wins_from_possible_win_starts = np.floor(n_possible_win_starts / n_shift) # Number of full windows we can get out of this set
    max_num_wins = n_full_wins_from_possible_win_starts + 1 # Add one more because the final sample in n_possible_win_starts can always be a sample (though the real start will likely be before this)
    num_wins_final = int(min(num_wins, max_num_wins)) if num_wins is not None else int(max_num_wins) # Use num_wins if provided and valid, otherwise use the maximum number of windows we can get
    
    window = get_window(win_type, n_win)
    norm = 2*np.ones((n_win // 2) + 1) # Multiply by two for real FFT (only using half the frequencies)
    norm[0] = 1  # ...except for the DC and
    norm[-1] = 1 # Nyquist frequencies
    norm /= np.sum(window)**2 # Squared sum of window coefficients for window normalization (just N for rectangle)
    
    # Divide the waveform into windows of length n_win, take magnitude of FFT of each
    powers_list = [
        norm*np.abs(rfft(wf[i*n_shift : i*n_shift + n_win]*window))**2 # Each new window has a start index that's n_shift more than the last
        for i in range(num_wins_final)
    ]

    # Average over all windows
    avg_powers = np.mean(powers_list, axis=0)
    return rfftfreq(n_win, 1/44100), 10 * np.log10(avg_powers)

def get_pg(wf, win_type='hann', dB=True, rescale=True):
    n_win=len(wf)
    window = get_window(win_type, n_win)
    norm = 2*np.ones((n_win // 2) + 1) # Multiply by two for real FFT (only using half the frequencies)
    norm[0] = 1  # ...except for the DC and
    norm[-1] = 1 # ...Nyquist frequencies
    norm /= np.sum(window)**2 # Squared sum of window coefficients for window normalization (just N for rectangle)
    
    # Rescale waveform
    if rescale:
        wf = lab_rescale(wf)
    
    return 10*np.log10(norm*np.abs(rfft(wf*window))**2)

def get_scipy_welch(wf, shift_prop=0.5, win_type='hann', rescale=True):
    n_win = 32768
    noverlap = int(shift_prop * n_win)
    if rescale:
        wf = lab_rescale(wf)

    return 10 * np.log10(welch(wf, fs=1, nperseg=n_win, noverlap=noverlap, window=win_type, scaling='spectrum', detrend=False)[1])
    

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

def process_wfs(df):
    """Take FFT of all raw waveforms in the dataframe and convert to dB SPL using Welch's Method"""
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

        db_SPL = get_welch_og_chris(wf)

        # Update the DataFrame directly
        df.at[idx, 'spectrum'] = db_SPL
        # Remove waveform for space
        df.at[idx, 'wf'] = []
        # No need to store freq ax, as these can be generated via rfftfreq(n_win, 1 / sr)
        
    return df