#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:49:25 2025

@author: Jorge Cabrera-Moreno
Postdoctoral Fellow
Evolutionary Cognition Group
Institute of Evolutionary Anthropology
University of Zurich
Switzerland

    Description: Python script that loads an audio file, plots its waveform, 
        spectrogram, and power spectral density, and then suggests additional 
        plots (MFCC and chroma features) for further insights into the audio signal.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import welch

import os
import random

def select_random_files(folder_path, number_of_files):
    # List all entries in the folder and filter to include only files
    all_entries = os.listdir(folder_path)
    files = [entry for entry in all_entries if os.path.isfile(os.path.join(folder_path, entry))]
    
    # Check if the requested number is more than available files
    if number_of_files > len(files):
        raise ValueError("Requested number of files exceeds the number of files in the folder.")
    
    # Randomly select the given number of files
    selected_files = random.sample(files, number_of_files)
    return selected_files

# Example usage:
if __name__ == "__main__":
    folder = "/Volumes/jcabreramoreno/Currently_Collected_Data/analisis_IRALL/GoNoGo/HearingThresholds/mxbi2/wol-duc/NoiseAnalysis/5kHz/Recordings"  # Replace with your folder path
    # num_files = 10  # Replace with the desired number of files to select

    # try:
    #     random_files = select_random_files(folder, num_files)
    #     print("Randomly selected files:", random_files)
    # except Exception as e:
        # print("Error:", e)
    
    random_files = ['mxbi2_20230317165050_wolfgang_correct.wav',
                    'mxbi2_20230313134946_wolfgang_correct.wav',
                    'mxbi2_20221020142906_wolfgang_correct.wav',
                    'mxbi2_20221020140919_wolfgang_correct.wav',
                    'mxbi2_20221007145004_wolfgang_correct.wav']

    for file in random_files:
        audio_path = os.path.join(folder, file)

        # # ---------------------------
        # # 1. Load the audio file
        # # ---------------------------
        # # Replace 'path/to/audio/file.wav' with your actual audio file path.
        # audio_path = '/Users/moreno/Documents/GitHub/VocalNets/AcousticCameraValidation/Code/exponential_sweep_1kHz_30kHz_10s.wav'
        # Load the audio with its native sampling rate (sr=None)
        y, sr = librosa.load(audio_path, sr=None)
        
        # # ---------------------------
        # # 2. Plot the Waveform
        # # ---------------------------
        # plt.figure(figsize=(10, 4))
        # # librosa.display.waveshow plots the amplitude over time.
        # librosa.display.waveshow(y, sr=sr)
        # plt.title('Waveform')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.show()
        
        # ---------------------------
        # 3. Plot the Spectrogram
        # ---------------------------
        plt.figure(figsize=(10, 4))
        # Compute the Short-Time Fourier Transform (STFT)
        S = librosa.stft(y)
        # Convert the amplitude to decibels for better visualization
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        # Display the spectrogram; x_axis 'time' and y_axis 'hz' labels the axes.
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
        plt.title('Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
    
    # # ---------------------------
    # # 4. Plot the Power Spectral Density (PSD)
    # # ---------------------------
    # # Use Welch's method to compute the PSD
    # frequencies, psd = welch(y, fs=sr, nperseg=1024)
    # plt.figure(figsize=(10, 4))
    # plt.semilogy(frequencies, psd)  # semilogy for logarithmic scale on y-axis
    # plt.title('Power Spectral Density (Welch)')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('PSD (V^2/Hz)')
    # plt.tight_layout()
    # plt.show()
    
    # # ---------------------------
    # # Additional Plots for More Insights
    # # ---------------------------
    
    # # 5. Mel-Frequency Cepstral Coefficients (MFCCs)
    # # MFCCs capture the timbral aspects of audio and are widely used in audio processing.
    # mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    # plt.title('MFCC')
    # plt.xlabel('Time (s)')
    # plt.ylabel('MFCC Coefficients')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    
    # # 6. Chroma Feature
    # # Chroma features represent the energy distribution across the 12 pitch classes.
    # chroma = librosa.feature.chroma_stft(y, sr=sr)
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
    # plt.title('Chroma Feature')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Pitch Class')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    
    # # 7. (Optional) Zero-Crossing Rate
    # # This metric indicates the rate at which the signal changes sign,
    # # which can be useful for understanding the noisiness of the signal.
    # zero_crossings = librosa.zero_crossings(y, pad=False)
    # print('Zero-crossing rate:', sum(zero_crossings))
