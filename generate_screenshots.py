import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from fourier_transform import Analyzer
from signal_processor import SignalProcessor
from signal_generator import SignalGenerator

def main():
    # 1. Generate base signal (Sine wave)
    gen = SignalGenerator(sample_rate=1000, duration=1.0)
    t, sig_base = gen.generate_sine(frequency=10, amplitude=1.0)
    
    # Add noise
    sig_noisy = SignalProcessor.add_noise(sig_base, noise_amplitude=0.3)
    
    # Process: Low-pass filter to remove noise
    sig_filtered = SignalProcessor.apply_lowpass_filter(sig_noisy, 1000, cutoff_freq=20.0)
    
    # FFT
    freqs_base, mag_base = Analyzer.perform_fft(sig_base, 1000)
    freqs_noisy, mag_noisy = Analyzer.perform_fft(sig_noisy, 1000)
    freqs_filtered, mag_filtered = Analyzer.perform_fft(sig_filtered, 1000)
    
    # --- Time Domain Plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, sig_noisy, label="Noisy Signal", color='lightgray', alpha=0.8)
    plt.plot(t, sig_base, label="Original Sine", color='blue', linestyle='dashed')
    plt.plot(t, sig_filtered, label="Filtered Signal", color='red')
    plt.title("Time Domain Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('time_domain.png')
    plt.close()
    
    # --- Frequency Domain Plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(freqs_noisy, mag_noisy, label="Noisy Signal FFT", color='lightgray', alpha=0.8)
    plt.plot(freqs_filtered, mag_filtered, label="Filtered Signal FFT", color='red')
    plt.xlim(0, 100) # zoom in on lower frequencies
    plt.title("Frequency Domain Analysis (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('frequency_domain.png')
    plt.close()

if __name__ == "__main__":
    main()
