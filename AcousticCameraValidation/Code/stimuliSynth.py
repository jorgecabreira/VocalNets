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

def generate_exponential_sweep(
    f_start=1000,    # start freq in Hz
    f_end=30000,     # end freq in Hz
    duration=10.0,   # total sweep duration in seconds
    samplerate=192000, 
    fade_time=0.02   # fade-in/out duration in seconds
):
    """
    Generate an exponential sine sweep from f_start to f_end over 'duration' seconds.
    Also applies a simple linear fade-in and fade-out of length 'fade_time'.
    Returns a NumPy array containing the sweep signal.
    """
    
    # Time vector
    t = np.linspace(0, duration, int(samplerate*duration), endpoint=False)

    # Exponential sweep formula:
    # phase(t) = 2π * ( (duration * f_start) / ln(f_end / f_start) ) * ( (f_end/f_start)^(t/duration) - 1 )
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

if __name__ == "__main__":
    samplerate = 192000
    sweep_signal = generate_exponential_sweep(
        f_start=1000, 
        f_end=30000,
        duration=10.0,
        samplerate=samplerate,
        fade_time=0.02  # 20 ms fade
    )
    
    # Write to WAV (32-bit float) – adjust filename/bit depth as needed
    sf.write("exponential_sweep_1kHz_30kHz_10s.wav", sweep_signal, samplerate, subtype='FLOAT')

