import numpy as np
from scipy.special import wofz

# Define profiles

# Lorentzian peak
def Lorentzian(x, x0, y0, amp, gamma):
    lorentzian = gamma**2 / ((x - x0)**2 + gamma**2)
    return y0 + amp * lorentzian # Note max height is exactly amp

# (Sampled) DTFT of a boxcar window with nperseg nonzero points. Note this is where the factor of "N" gets introduced that needs scaling out in the IDFT
def boxcar_DTFT(w, N, fs):
    def func(w):
        return np.exp(-1j * N * w / 2) * (np.sin((N + 1) * w / 2)) / (np.sin(w / 2))
    # Handle the case where w = 0
    if np.any(w == 0):
        center_idx = np.where(w == 0)[0][0]
        print("hello")
        result = np.empty_like(w, dtype=complex)
        result[0:center_idx] = func(w[0:center_idx])
        result[center_idx + 1:] = func(w[center_idx + 1:])
        result[center_idx] = N
    else:
        result = func(w)
    return result

def damped_sinusoid_DTFT(f, f0, A, gamma, fs):
    # Convert to normalized radial frequency
    w = 2 * np.pi * f / fs
    w0 = 2 * np.pi * f0 / fs
    # Compute the two terms in the DTFT expression
    term1 = 1 / (1 - np.exp(-(gamma + 1j * (w - w0))))
    term2 = 1 / (1 - np.exp(-(gamma + 1j * (w + w0))))

    # Combine the terms with the amplitude scaling
    return (A / 2) * (term1 + term2)

# Convolution of Lorentz with the boxcar
def Lorentzian_conv(f, f0, y0, A, gamma, nperseg, fs):
    """
    Compute the convolution of a Lorentzian peak (technically, the damped sinusoid DTFT) with a boxcar function.
    
    Parameters:
        f (array): Positive half of FFT frequencies (starting from 0).
        f0 (float): Center of the Lorentzian peaks.
        y0 (float): Offset of the Lorentzian peaks.
        amp (float): Amplitude of the Lorentzian peaks.
        gamma (float): Half-width at half-maximum (HWHM) of the Lorentzian peas.
        nperseg (int): Length of the boxcar function.
        fs (float): Sampling frequency.
    
    Returns:
        array: Convolution result corresponding to the original indices of x, with the peak preserved.
    """
    # Compute the Lorentzian function peaks (positive and negative)
        # Is this bleeding of the positive freq peak into the negative freqs (and vice versa) correct?
        # YES
    X =  damped_sinusoid_DTFT(f, f0, A, gamma, fs)
    # Compute the boxcar DTFT 
    boxcar = boxcar_DTFT(f, nperseg, fs)
    # Circular convolution via FFTs
    result = np.real(np.fft.ifft(np.fft.fft(X)*np.fft.fft(boxcar))) + y0
    return np.abs(result)


def Voigt(x, x0, y0, amp, gamma, sigma):
    z = ((x - x0) + 1j*gamma)/(sigma*np.sqrt(2))
    voigt_profile = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return y0 + amp * voigt_profile / voigt_profile.max() # Normalize so max height = amp

def Gauss(x, x0, y0, amp, sigma):
    return y0 + amp * np.exp(-(x - x0)**2 / (2 * sigma**2)) # Already true that max height = amp

def get_gauss_hwhm(sigma):
    return sigma*np.sqrt(2*np.log(2))

def get_voigt_hwhm(gamma, sigma):
    f_G = 2*get_gauss_hwhm(sigma)
    f_L = 2*gamma
    return (0.5346*f_L + np.sqrt(0.2166*f_L**2 + f_G**2))/2