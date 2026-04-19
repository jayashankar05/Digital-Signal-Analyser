import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from signal_generator import SignalGenerator
from signal_processor import SignalProcessor
from fourier_transform import Analyzer

class DigitalSignalAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Signal Analyzer")
        self.root.geometry("1200x800")
        
        self.generator = SignalGenerator()
        
        self.t = None
        self.original_signal = None
        self.processed_signal = None
        
        self._create_widgets()
        
    def _create_widgets(self):
        # Control Panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # --- Signal Generation Parameters ---
        gen_lf = ttk.LabelFrame(control_frame, text="Signal Generation", padding="10")
        gen_lf.pack(fill=tk.X, pady=5)
        
        ttk.Label(gen_lf, text="Type:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.sig_type_var = tk.StringVar(value="Sine")
        ttk.Combobox(gen_lf, textvariable=self.sig_type_var, values=["Sine", "Square", "Noise"], state="readonly").grid(row=0, column=1, pady=2)
        
        ttk.Label(gen_lf, text="Frequency (Hz):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.freq_var = tk.DoubleVar(value=10.0)
        ttk.Entry(gen_lf, textvariable=self.freq_var).grid(row=1, column=1, pady=2)
        
        ttk.Label(gen_lf, text="Amplitude:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.amp_var = tk.DoubleVar(value=1.0)
        ttk.Entry(gen_lf, textvariable=self.amp_var).grid(row=2, column=1, pady=2)
        
        ttk.Label(gen_lf, text="Phase (rad):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.phase_var = tk.DoubleVar(value=0.0)
        ttk.Entry(gen_lf, textvariable=self.phase_var).grid(row=3, column=1, pady=2)
        
        ttk.Label(gen_lf, text="Sample Rate (Hz):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.sr_var = tk.IntVar(value=1000)
        ttk.Entry(gen_lf, textvariable=self.sr_var).grid(row=4, column=1, pady=2)
        
        ttk.Label(gen_lf, text="Duration (s):").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.dur_var = tk.DoubleVar(value=1.0)
        ttk.Entry(gen_lf, textvariable=self.dur_var).grid(row=5, column=1, pady=2)
        
        ttk.Button(gen_lf, text="Generate Base Signal", command=self.generate_signal).grid(row=6, column=0, pady=10)
        ttk.Button(gen_lf, text="Load .WAV", command=self.load_wav_file).grid(row=6, column=1, pady=10)
        
        # --- Processing Parameters ---
        proc_lf = ttk.LabelFrame(control_frame, text="Signal Processing", padding="10")
        proc_lf.pack(fill=tk.X, pady=5)
        
        self.proc_var = tk.StringVar(value="None")
        
        ttk.Radiobutton(proc_lf, text="None", variable=self.proc_var, value="None").grid(row=0, column=0, sticky=tk.W, columnspan=2)
        
        ttk.Radiobutton(proc_lf, text="Add Noise", variable=self.proc_var, value="Noise").grid(row=1, column=0, sticky=tk.W)
        self.noise_amp_var = tk.DoubleVar(value=0.5)
        ttk.Entry(proc_lf, textvariable=self.noise_amp_var, width=10).grid(row=1, column=1)
        
        ttk.Radiobutton(proc_lf, text="Smooth (Moving Avg)", variable=self.proc_var, value="Smooth").grid(row=2, column=0, sticky=tk.W)
        self.window_size_var = tk.IntVar(value=10)
        ttk.Entry(proc_lf, textvariable=self.window_size_var, width=10).grid(row=2, column=1)
        
        ttk.Radiobutton(proc_lf, text="Low-pass Filter", variable=self.proc_var, value="Lowpass").grid(row=3, column=0, sticky=tk.W)
        self.lp_cutoff_var = tk.DoubleVar(value=50.0)
        ttk.Entry(proc_lf, textvariable=self.lp_cutoff_var, width=10).grid(row=3, column=1)
        
        ttk.Radiobutton(proc_lf, text="High-pass Filter", variable=self.proc_var, value="Highpass").grid(row=4, column=0, sticky=tk.W)
        self.hp_cutoff_var = tk.DoubleVar(value=5.0)
        ttk.Entry(proc_lf, textvariable=self.hp_cutoff_var, width=10).grid(row=4, column=1)
        
        ttk.Button(proc_lf, text="Apply Processing", command=self.apply_processing).grid(row=5, column=0, columnspan=2, pady=10)
        
        # Plots Panel
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, (self.ax_time, self.ax_freq) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def generate_signal(self):
        try:
            sr = self.sr_var.get()
            dur = self.dur_var.get()
            freq = self.freq_var.get()
            amp = self.amp_var.get()
            sig_type = self.sig_type_var.get()
            
            self.generator.set_parameters(sr, dur)
            
            if sig_type == "Sine":
                self.t, self.original_signal = self.generator.generate_sine(freq, amp, self.phase_var.get())
            elif sig_type == "Square":
                self.t, self.original_signal = self.generator.generate_square(freq, amp)
            elif sig_type == "Noise":
                self.t, self.original_signal = self.generator.generate_noise(amp)
                
            self.processed_signal = None
            self.update_plots()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate signal: {e}")

    def load_wav_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not file_path:
            return
            
        try:
            from scipy.io import wavfile
            sample_rate, data = wavfile.read(file_path)
            
            # Convert to mono if stereo
            if len(data.shape) == 2:
                data = data.mean(axis=1)
                
            # Normalize audio
            if data.dtype == np.int16:
                data = data / 32768.0
            elif data.dtype == np.int32:
                data = data / 2147483648.0
                
            self.sr_var.set(sample_rate)
            duration = len(data) / sample_rate
            self.dur_var.set(duration)
            
            self.t = np.linspace(0, duration, len(data), endpoint=False)
            self.original_signal = data
            self.processed_signal = None
            
            self.sig_type_var.set("WAV File")
            self.update_plots()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load WAV file: {e}")
            
    def apply_processing(self):
        if self.original_signal is None:
            messagebox.showwarning("Warning", "Please generate a signal first.")
            return
            
        try:
            proc_type = self.proc_var.get()
            sr = self.sr_var.get()
            
            if proc_type == "None":
                self.processed_signal = None
            elif proc_type == "Noise":
                self.processed_signal = SignalProcessor.add_noise(self.original_signal, self.noise_amp_var.get())
            elif proc_type == "Smooth":
                self.processed_signal = SignalProcessor.smooth_signal(self.original_signal, self.window_size_var.get())
            elif proc_type == "Lowpass":
                self.processed_signal = SignalProcessor.apply_lowpass_filter(self.original_signal, sr, self.lp_cutoff_var.get())
            elif proc_type == "Highpass":
                self.processed_signal = SignalProcessor.apply_highpass_filter(self.original_signal, sr, self.hp_cutoff_var.get())
                
            self.update_plots()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process signal: {e}")
            
    def update_plots(self):
        self.ax_time.clear()
        self.ax_freq.clear()
        
        if self.original_signal is not None:
            # Time Domain Plot
            self.ax_time.plot(self.t, self.original_signal, label="Original", color='blue', alpha=0.7)
            self.ax_time.set_title("Time Domain")
            self.ax_time.set_xlabel("Time (s)")
            self.ax_time.set_ylabel("Amplitude")
            
            # Frequency Domain Plot
            sr = self.sr_var.get()
            freqs, mag = Analyzer.perform_fft(self.original_signal, sr)
            self.ax_freq.plot(freqs, mag, label="Original", color='blue', alpha=0.7)
            
            # Processed Signal
            if self.processed_signal is not None:
                self.ax_time.plot(self.t, self.processed_signal, label="Processed", color='red', alpha=0.7)
                
                freqs_p, mag_p = Analyzer.perform_fft(self.processed_signal, sr)
                self.ax_freq.plot(freqs_p, mag_p, label="Processed", color='red', alpha=0.7)
                
            self.ax_time.legend(loc='upper right')
            
            self.ax_freq.set_title("Frequency Domain (FFT)")
            self.ax_freq.set_xlabel("Frequency (Hz)")
            self.ax_freq.set_ylabel("Magnitude")
            self.ax_freq.legend(loc='upper right')
            self.ax_freq.grid(True)
            self.ax_time.grid(True)
            
        self.canvas.draw()
