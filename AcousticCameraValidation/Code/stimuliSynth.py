#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:36:53 2025

@author: Jorge Cabrera-Moreno
Postdoctoral Fellow
Evolutionary Cognition Group
Institute of Evolutionary Anthropology
University of Zurich
Switzerland

    Description: The code synthesizes acoustic stimuli to be tested in the
        acoustic chamber. They will serve as validation stimuli to observe the
        performace of the acoustic camera on capturing sound sources.
"""

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

def generate_exponential_sweep(
    f_start=1000,    # start freq in Hz
    f_end=30000,     # end freq in Hz
    duration=10.0,   # total sweep duration in seconds
    samplerate=192000,
    fade_time=0.02   # fade-in/out duration in seconds
):
    """
    Generate an exponential (log) sine sweep from f_start to f_end over 'duration' seconds.
    Applies a linear fade-in and fade-out of length 'fade_time'.
    Returns a NumPy array containing the sweep signal.
    """
    
    # Time vector
    t = np.linspace(0, duration, int(samplerate*duration), endpoint=False)

    # Exponential sweep formula
    # phase(t) = 2π * ( (duration*f_start)/ln(f_end/f_start) ) * [ (f_end/f_start)^(t/duration) - 1 ]
    K = duration * f_start / np.log(f_end / f_start)
    L = np.log(f_end / f_start)
    phase = 2 * np.pi * K * (np.exp((L * t) / duration) - 1)
    
    sweep = np.sin(phase)

    # Apply linear fade-in and fade-out
    fade_samples = int(fade_time * samplerate)
    if fade_samples > 0 and 2*fade_samples < len(sweep):
        # Fade-in ramp from 0 to 1
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        # Fade-out ramp from 1 to 0
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        sweep[:fade_samples] *= fade_in
        sweep[-fade_samples:] *= fade_out
    
    return sweep


def generate_bandlimited_noise(
    duration=0.1,        # duration in seconds (e.g. 0.1s = 100 ms)
    samplerate=192000, 
    f_low=5000,          # lower cutoff frequency (Hz)
    f_high=30000,        # upper cutoff frequency (Hz)
    fade_time=0.01,      # fade-in/out duration in seconds
    filter_order=8       # order of Butterworth filter
):
    """
    Generate a band-limited noise burst of given 'duration' from f_low to f_high.
    Uses a Butterworth bandpass filter. Also applies a short linear fade-in/out 
    to avoid clicks. Returns a NumPy array of the filtered noise.
    """

    # Number of samples
    N = int(duration * samplerate)

    # Create white noise
    noise = np.random.normal(0, 1, N)

    # Design a bandpass Butterworth filter
    sos = butter(filter_order, [f_low, f_high], btype='band', fs=samplerate, output='sos')

    # Apply filter to the noise
    filtered_noise = sosfilt(sos, noise)

    # Apply fade-in/fade-out
    fade_samples = int(fade_time * samplerate)
    if fade_samples > 0 and 2 * fade_samples < len(filtered_noise):
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        filtered_noise[:fade_samples] *= fade_in
        filtered_noise[-fade_samples:] *= fade_out

    # Normalize to prevent clipping (optional)
    peak = np.max(np.abs(filtered_noise))
    if peak > 0:
        filtered_noise = 0.9 * (filtered_noise / peak)  # -1..1 range

    return filtered_noise


if __name__ == "__main__":
    samplerate = 192000
    
    ############################
    # 1) Example: Generate Sweep
    ############################
    fStart = 1000 # Frequency start
    fEnd = 30000 # Frequency end
    duration = 10.0
    
    sweep_signal = generate_exponential_sweep(
        f_start=fStart,
        f_end=fEnd,
        duration=duration,
        samplerate=samplerate,
        fade_time=0.02  # 20 ms fade
    )
    filename = f"exponential_sweep_{fStart/1000}kHz-{fEnd/1000}kHz_{duration}s.wav"
    sf.write(filename, sweep_signal, samplerate, subtype='FLOAT')
    print(f"Saved {filename}")

    #####################################
    # 2) Example: Generate Noise Bursts
    #####################################
    durations_ms = [100, 250, 500] # ms
    fLow = 1000 # Lowest frequency
    fHigh = 30000 # Highest frequency
    for d_ms in durations_ms:
        duration_sec = d_ms / 1000.0
        noise_burst = generate_bandlimited_noise(
            duration=duration_sec,
            samplerate=samplerate,
            f_low=fLow,      
            f_high=fHigh,   
            fade_time=0.01,  # 10 ms fade
            filter_order=8
        )
        filename = f"bandlimited_noise_{fLow/1000}kHz-{fHigh/1000}kHz_{d_ms}ms.wav"
        sf.write(filename, noise_burst, samplerate, subtype='FLOAT')
        print(f"Saved {filename}")
