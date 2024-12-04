import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.io
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
import scipy.signal.windows as wins

def get_mags(wf, sr, t_win, num_wins=None, hann=False, wfft=None, freq_ax=None, dict=False, norm="forward"):
  """ Gets the magnitudes of the given waveform, averaged over windows (with the given window size)

  Parameters
  ------------
      wf: array
        waveform input array
      sr: int
        sample rate of waveform
      t_win: float
        length (in time) of each window
      num_wins: int, Optional
        Used in get_wfft; if this isn't passed, then just gets the maximum number of windows of the given size
      hann: bool, Optional
        Applies a Hann window before taking fft
      wfft: any, Optional
        If you want to avoid recalculating the windowed fft, pass it in here!
      freq_ax: any, Optional
        We have to also pass in the freq_ax for the wfft (either pass in both or neither!)
      dict: bool, Optional
        Defaults to only returning the PSD averaged over all windows; if this is enabled, then a dictionary is returned with keys:
        "mags", "freq_ax", "win_mags"
      norm: string, Optional
        Whether Scipy normalizes in the forward or backward FFT direction
  """
  # make sure we either have both or neither
  if (wfft is None and freq_ax is not None) or (wfft is not None and freq_ax is None):
    raise Exception("We need both wfft and freq_ax (or neither)!")
  
  # if you passed the wfft and freq_ax in then we'll skip over this
  if wfft is None:
    d = get_wfft(wf=wf, sr=sr, t_win=t_win, num_wins=num_wins, hann=hann, norm=norm)
    wfft = d["wfft"]
    freq_ax = d["freq_ax"]

  
  # calculate necessary params from the wfft
  wfft_size = np.shape(wfft)
  num_wins = wfft_size[0]
  num_freq_pts = wfft_size[1]
  
  # initialize array
  win_mags = np.zeros((num_wins, num_freq_pts))

  # get magnitudes for each window
  for win in range(num_wins):
    win_mags[win, :] = np.abs(wfft[win, :])
    
  # average over all windows
  mags = np.mean(win_mags, 0)

  if not dict:
    return mags
  else:
    return {  
      "mags" : mags,
      "freq_ax" : freq_ax,
      "win_mags" : win_mags
      }

def get_wfft(wf, sr, t_win, t_shift=None, num_wins=None, hann=False, norm="backward"):
  """ Returns a dict with the windowed fft and associated freq ax of the given waveform

  Parameters
  ------------
      wf: array
        waveform input array
      sr: int
        sample rate of waveform
      t_win: float
        length (in time) of each window
      t_shift: float
        length (in time) between the start of successive windows
      num_wins: int, Optional
        If this isn't passed, then just get the maximum number of windows of the given size
      hann: bool, Optional
        Applied a hanning window before the FFT
      norm: string, Optional
        Whether Scipy normalizes in the forward or backward FFT direction
  """
  
  # if you didn't pass in t_shift we'll assume you want no overlap - each new window starts at the end of the last!
  if t_shift is None:
    t_shift=t_win
  
  # calculate the number of samples in the window
  n_win = int(t_win*sr)

  # and the number of samples to shift
  n_shift = int(t_shift*sr)

  # get sample_spacing
  sample_spacing = 1/sr

  # first, get the last index of the waveform
  final_wf_index = len(wf) - 1
    # - 1 is because of zero-indexing!
  # next, we get what we would be the largest potential win_start_index
  final_win_start_index = final_wf_index - (n_win-1)
    # start at the final_wf_index. we need to collect n_win points. this final index is our first one, and then we need n_win - 1 more. 
    # So we march back n_win-1 points, and then THAT is the last_potential_win_start_index!
  win_start_indices = np.arange(0, final_win_start_index + 1, n_shift)
    # the + 1 here is because np.arange won't ever include the "stop" argument in the output array... but it could include (stop - 1) which is just our final_win_start_index!

  # if number of windows is passed in, we make sure it's less than the length of win_start_indices
  if num_wins is not None:
    if num_wins > len(win_start_indices):
      raise Exception("That's more windows than we can manage! Decrease num_wins!")
  else:
    # if no num_wins is passed in, we'll just use the max number of windows
    num_wins = len(win_start_indices)

  windowed_wf = np.zeros((num_wins, n_win))
  for k in range(num_wins):
    win_start = win_start_indices[k]
    win_end = win_start + n_win
    # grab the (real part of the) waveform in this window
    windowed_wf[k, :] = wf[win_start:win_end].real
    # note this grabs the wf at indices win_start, win_start+1, ..., win_end-1
      # if there are 4 samples and t_win=t_shift=1 and SR=1, then n_win=2, n_shift=1 and
      # Thus the first window will be samples 0 and 1, the next 1 and 2...

  # Now we do the ffts!

  # get frequency axis 
  freq_ax = rfftfreq(n_win, sample_spacing)
  num_freq_pts = len(freq_ax)
  
  # initialize windowed fft array
  wfft = np.zeros((num_wins, num_freq_pts), dtype=complex)
  # get ffts (optionally apply the Hann window first) 
    # norm=forward applies the normalizing 1/n_win factor 
  if hann:
    for k in range(num_wins):
      # wfft[k, :] = rfft(windowed_wf[k, :]*np.hanning(n_win), norm="forward")
      wfft[k, :] = rfft(windowed_wf[k, :]*wins.hann(n_win, sym=True), norm=norm)
      # w = rfft(windowed_wf[k, :], norm="forward")
      # wfft[k, :] = np.convolve(w, [-1/4, 1/2, -1/4], mode="same")
  else:
    for k in range(num_wins):
      wfft[k, :] = rfft(windowed_wf[k, :], norm=norm)
    
  return {  
    "wfft" : wfft,
    "freq_ax" : freq_ax,
    "win_start_indices" : win_start_indices
    }

def smooth_data(y, window_size=5):
    """
    Smooth the data using a simple moving average.
    
    Parameters:
    - y: array-like, y values to smooth
    - window_size: int, size of the moving average window
    
    Returns:
    - smoothed_y: array, smoothed y values
    """
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')

def nth_highest_local_maximum(x, y, n, smoothing_window=5, min_distance=10, min_prominence=0.000001):
    """
    Find the nth highest local maximum of the given curve with smoothing and filtering.
    
    Parameters:
    - x: array-like, x values of the curve
    - y: array-like, y values of the curve
    - n: int, the rank of the local maximum to find (1 for the highest, 2 for the second highest, etc.)
    - smoothing_window: int, size of the moving average window for smoothing
    - min_distance: int, minimum distance between peaks
    - min_prominence: float, minimum prominence of peaks
    
    Returns:
    - nth_max_x: float, x value of the nth highest local maximum
    - nth_max_y: float, y value of the nth highest local maximum
    """
    # Smooth the y values
    smoothed_y = smooth_data(y, window_size=smoothing_window)
    
    # Adjust x values to match the length of smoothed y
    adjusted_x = x[(smoothing_window - 1) // 2: -(smoothing_window // 2)]
    
    # Find local maxima using scipy's find_peaks with distance and prominence
    peaks, properties = find_peaks(smoothed_y, distance=min_distance, prominence=min_prominence)

    # Extract the y values of the local maxima
    peak_values = smoothed_y[peaks]
    
    # Combine peak indices and values, and sort them by values
    sorted_peaks = sorted(zip(peaks, peak_values), key=lambda x: x[1], reverse=True)
    
    # print(sorted_peaks)

    # Check if there are enough peaks to return the nth highest
    if n > len(sorted_peaks):
        raise ValueError(f"Requested {n}th highest local maximum, but only {len(sorted_peaks)} local maxima found.")

    # Get the nth highest peak
    nth_peak = sorted_peaks[n] #note this starts with the "0th" biggest peak
        
    # Extract x and y values of the nth peak
    nth_max_x = adjusted_x[nth_peak[0]]
    nth_max_y = nth_peak[1]

    return nth_max_x, nth_max_y

# Define the Lorentzian function
def lorentzian(x, A, x0, gamma):
    return A / ((x - x0)**2 + gamma**2)

# Define the combined Lorentzian function
def combined_lorentzian(x, *params):
    n = len(params) // 3  # Number of Lorentzian components
    y = np.zeros_like(x)
    for i in range(n):
        A = params[3 * i]
        x0 = params[3 * i + 1]
        gamma = params[3 * i + 2]
        y += lorentzian(x, A, x0, gamma)
    return y


# LOAD DATA
filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"
# filename = 'TH14RearwaveformSOAE.mat'
filename = 'ACsb24rearSOAEwfA1'
mat = scipy.io.loadmat(filepath + filename)
wf = np.squeeze(mat['wf'])
sr = 44100

# parameters for fourier transform
trunc_fraction = 1 # divide wf length by this number
t_win = 10 # window size for FT
freq_min = 1000 # minimum frequency to keep
freq_max = 5000 # max frequency to keep

# parameters for identifying peaks
min_peak_distance_hertz = 50 # in hertz
smoothing_window_hertz = 10 # we'll do some smoothing to make it easier to pick out the peaks (smoothing window size in hertz)

# parameters for the lorentz-engine
min_lorentzians = 10 # minimum number of lorentz functions to fit
max_lorentzians = 10 # maximum number of lorentz functions to fit
rounds_per_num_lorentzians = 2 # how many (ideally successful) rounds we do for each number of lorentzians
max_retries = 3   # Maximum number of retries per round (when it doesn't converge) before we just move on


# define some things
wf = wf[0:int(len(wf) / trunc_fraction)]
d = get_mags(wf, sr, t_win=t_win, dict=True)

x_data = d["freq_ax"]
y_data = d["mags"]

min_i = np.where(x_data == freq_min)[0][0] 
max_i = np.where(x_data == freq_max)[0][0]
x_data = x_data[min_i:max_i]
y_data = y_data[min_i:max_i]

best_aic = float('inf')
best_params = None
best_n = 0


# convert from hertz to samples (each freq bin is 1/t_win wide, so for each hertz we need t_win # bins to fill it out)
min_peak_sample_distance = int(min_peak_distance_hertz * t_win)
smoothing_Window = int(smoothing_window_hertz * t_win)


# Attempt fitting for a varying number of Lorentzians
for n in range(min_lorentzians, max_lorentzians + 1):
    print(f"{n} Lorentzians")
    # Try each each one until you've successfully (or failed enough retried) found tries_per_num_lorentzians for each # of lorentzians 
    for k in range(1, rounds_per_num_lorentzians + 1):
        print(f"Number {k}")
        fitting_success = False
        attempt = 0
        # for each round, if we fail to converge we'll try again until we hit max_retries
        while not fitting_success and attempt < max_retries:
            initial_guess = []
            
            for i in range(n):
                # A = np.random.uniform(0.1, 2.0)  # Random amplitude
                # x0 = np.random.uniform(0, 100)  # Random center
                # gamma = np.random.uniform(0.1, 5.0)  # Random width
                
                # center at the local max
                x0, y0 = nth_highest_local_maximum(x_data, y_data, i, min_distance=min_peak_sample_distance)
                # print(f"Max #{i} is at {x0}")
                
                # this seemed to be a good width
                gamma = 10
                
                initial_guess.extend([y0, x0, gamma])
            
            # Attempt to fit the combined Lorentzian model to the data
            try:
                params, covariance = curve_fit(combined_lorentzian, x_data, y_data, p0=initial_guess)
                fitting_success = True
                y_fit = combined_lorentzian(x_data, *params)
                print(f"Fitting succeeded for {n} Lorentzians with parameters: {params}")
                residuals = y_data - y_fit
                rss = np.sum(residuals**2)  # Residual sum of squares
                aic = 2 * len(params) + len(y_data) * np.log(rss / len(y_data))  # AIC calculation
                # Update best AIC and parameters
                if aic < best_aic:
                    best_aic = aic
                    best_params = params
                    best_n = n
            except RuntimeError as e:
                print(f"Fitting failed for {n} Lorentzians on attempt {attempt + 1}: {e}")
                attempt += 1
                continue
            except Exception as e:
                print(f"An unexpected error occurred for {n} Lorentzians on attempt {attempt + 1}: {e}")
                attempt += 1
                continue

        if fitting_success:
            y_best_fit = combined_lorentzian(x_data, *best_params)
        else:
            print(f"Failed to fit {n} Lorentzians after {max_retries} attempts.")
        
# Print the best fit parameters
print("Best fit with {} Lorentzians:".format(best_n))
for i in range(best_n):
    print(f"Lorentzian {i + 1}: A={best_params[3 * i]:.3f}, x0={best_params[3 * i + 1]:.3f}, gamma={best_params[3 * i + 2]:.3f}")

# Generate fitted data for plotting
y_best_fit = combined_lorentzian(x_data, *best_params)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data', color='blue', s=10)
plt.plot(x_data, y_best_fit, label='Best Fitted Combined Lorentzian', color='red')
plt.title(f'Fitting {best_n} Lorentzian Functions')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid()
plt.show()
