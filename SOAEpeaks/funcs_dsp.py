import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch, get_window

# import sys
# import os

# # Add the path to the folder containing the module
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'other_folder')))

# from helper import my_function

# print(my_function())


def lab_rescale(input):
    # Define constants that allow us to correct for lab setup
    amp_factor = 0.01  # Correct for amplifier gain
    # mic_factor = 10**(-6)  # Corresponds to the mic level of 1 microvolt
    mic_factor = 1
    rms_factor = np.sqrt(2)  # Converts to RMS value
    
    # Rescale input (either wf or magnitudes, NOT powers!)
    return input * amp_factor / (mic_factor * rms_factor)

def my_welch(wf, fs, nperseg, win_type='boxcar', nfft=None, num_wins=None, overlap=0.5, scaling='density'):
    if nfft is None:
        nfft = nperseg
    n_shift = int(nperseg*overlap)

    
    n_possible_win_starts = len(wf) - nperseg # Number of samples that could be a start of a sample
    n_full_wins_from_possible_win_starts = np.floor(n_possible_win_starts / n_shift) # Number of full windows we can get out of this set
    max_num_wins = n_full_wins_from_possible_win_starts + 1 # Add one more because the final sample in n_possible_win_starts can always be a sample (though the real start will likely be before this)
    num_wins_final = int(min(num_wins, max_num_wins)) if num_wins is not None else int(max_num_wins) # Use num_wins if provided and valid, otherwise use the maximum number of windows we can get
    
    window = get_window(win_type, nperseg)
    scale_factors = 2*np.ones((nfft // 2) + 1) # Multiply by two for real FFT (only using half the frequencies)
    scale_factors[0] = 1  # ...except for the DC and
    scale_factors[-1] = 1 # ...Nyquist frequencies
    
    if scaling == 'spectrum':
        scale_factors /= np.sum(window)**2 # (Just N**2 for rectangle)
    elif scaling == 'density':
        scale_factors /= (np.sum(window**2) * fs) # (Just N * fs = N^2 * bin_wdith for rectangle)
        
    
    # Divide the waveform into windows of length n_win, take magnitude of FFT of each
    powers_list = [
        scale_factors*np.abs(rfft(np.pad(wf[i*n_shift : i*n_shift + nperseg]*window, (0, nfft-nperseg), 'constant')))**2 # Each new window has a start index that's n_shift more than the last
        for i in range(num_wins_final)
    ]

    # Average over all windows
    return rfftfreq(nfft, 1/fs), np.mean(powers_list, axis=0)

def get_welch_og_chris(wf, rescale=True, dB=True):
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
        
    if dB:
        avg_mags = 20 * np.log10(avg_mags) - 20*np.log10(20*10**-6)
        
    return avg_mags
    
def get_welch(wf, shift_prop=0.5, win_type='hann', n_win=32768, num_wins=None, dB=True, rescale=True, zero_pad=1):
    # Define window length
    n_shift = int(n_win*shift_prop)
    n_padded = int(n_win * zero_pad)  # New variable for padded length (e.g., zero_pad = 4 for 4x zero-padding)

    
    # Rescale waveform
    if rescale:
        wf = lab_rescale(wf)
    
    n_possible_win_starts = len(wf) - n_win # Number of samples that could be a start of a sample
    n_full_wins_from_possible_win_starts = np.floor(n_possible_win_starts / n_shift) # Number of full windows we can get out of this set
    max_num_wins = n_full_wins_from_possible_win_starts + 1 # Add one more because the final sample in n_possible_win_starts can always be a sample (though the real start will likely be before this)
    num_wins_final = int(min(num_wins, max_num_wins)) if num_wins is not None else int(max_num_wins) # Use num_wins if provided and valid, otherwise use the maximum number of windows we can get
    
    window = get_window(win_type, n_win)
    norm = 2*np.ones((n_padded // 2) + 1) # Multiply by two for real FFT (only using half the frequencies)
    norm[0] = 1  # ...except for the DC and
    norm[-1] = 1 # Nyquist frequencies
    norm /= np.sum(window)**2 # Squared sum of window coefficients for window normalization (just N for rectangle)
    
    # Divide the waveform into windows of length n_win, take magnitude of FFT of each
    powers_list = [
        norm*np.abs(rfft(np.pad(
            wf[i * n_shift : i * n_shift + n_win] # Note each new window has a start index that's n_shift more than the last
            * window, # Multiply by window (hann, boxcar, etc) BEFORE zero padding
                              (0, n_padded - n_win), 'constant') # Pad with n_padded - n_win (constant) zeros so total is n_padded
                         ))**2 # Square to take the power
        for i in range(num_wins_final) # Do this for all windows
    ]

    # Average over all windows
    avg_powers = np.mean(powers_list, axis=0)
    
    if dB:
        avg_powers = 10 * np.log10(avg_powers) - 20*np.log10(20*10**-6)
    
    return rfftfreq(n_padded, 1/44100), avg_powers

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
    
    pg = norm*np.abs(rfft(wf*window))**2
    
    if dB:
        pg = 10*np.log10(pg) - 20*np.log10(20*10**-6)
    
    return pg

def get_scipy_welch(wf, shift_prop=0.5, n_win=32768, win_type='hann', rescale=True, dB=True):
    noverlap = int(shift_prop * n_win)
    if rescale:
        wf = lab_rescale(wf)
        
    f, w = welch(wf, fs=44100, nperseg=n_win, noverlap=noverlap, window=win_type, scaling='spectrum', detrend=False)
    
    if dB:
        w = 10 * np.log10(w) - 20*np.log10(20*10**-6)

    return f, w
    