import numpy as np
from scipy import signal

class SignalProcessor:
    """Class to perform processing operations on signals."""

    @staticmethod
    def add_noise(y, noise_amplitude=0.1):
        """Adds white Gaussian noise to a signal."""
        noise = np.random.normal(0, noise_amplitude, len(y))
        return y + noise

    @staticmethod
    def smooth_signal(y, window_size=5):
        """Smooths a signal using a moving average filter."""
        if window_size < 1:
            return y
        window = np.ones(window_size) / window_size
        return np.convolve(y, window, mode='same')

    @staticmethod
    def apply_lowpass_filter(y, sample_rate, cutoff_freq, order=4):
        """Applies a Butterworth low-pass filter."""
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        # Ensure valid cutoff frequency
        if normal_cutoff >= 1.0 or normal_cutoff <= 0.0:
            return y
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, y)

    @staticmethod
    def apply_highpass_filter(y, sample_rate, cutoff_freq, order=4):
        """Applies a Butterworth high-pass filter."""
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        # Ensure valid cutoff frequency
        if normal_cutoff >= 1.0 or normal_cutoff <= 0.0:
            return y
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, y)
