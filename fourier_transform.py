import numpy as np

class Analyzer:
    """Class to perform frequency domain analysis."""

    @staticmethod
    def perform_fft(y, sample_rate):
        """
        Computes the Fast Fourier Transform of a signal.
        Returns the frequency axis and the magnitude spectrum.
        """
        n = len(y)
        # Compute FFT
        yf = np.fft.fft(y)
        # Compute corresponding frequencies
        xf = np.fft.fftfreq(n, 1 / sample_rate)
        
        # We usually only care about the positive frequencies
        half_n = n // 2
        
        # Calculate magnitude spectrum and normalize
        magnitude = np.abs(yf[:half_n]) / (n / 2)
        # Adjust DC component (0 Hz)
        if len(magnitude) > 0:
            magnitude[0] = magnitude[0] / 2
            
        freqs = xf[:half_n]
        
        return freqs, magnitude
