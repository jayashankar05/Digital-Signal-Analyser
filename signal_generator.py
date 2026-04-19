import numpy as np
from scipy import signal

class SignalGenerator:
    """Class to generate different types of digital signals."""
    
    def __init__(self, sample_rate=1000, duration=1.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    def set_parameters(self, sample_rate, duration):
        """Update sample rate and duration, regenerating time axis."""
        self.sample_rate = sample_rate
        self.duration = duration
        self.t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    def generate_sine(self, frequency, amplitude=1.0, phase=0.0):
        """Generates a sine wave."""
        return self.t, amplitude * np.sin(2 * np.pi * frequency * self.t + phase)

    def generate_square(self, frequency, amplitude=1.0, duty_cycle=0.5):
        """Generates a square wave."""
        return self.t, amplitude * signal.square(2 * np.pi * frequency * self.t, duty=duty_cycle)

    def generate_noise(self, amplitude=1.0):
        """Generates random white noise."""
        return self.t, amplitude * np.random.normal(0, 1, len(self.t))
