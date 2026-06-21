"""
fingerprint.py

Audio Fingerprinting Module for EE200 Q3B

This module:
1. Loads audio
2. Computes an STFT spectrogram
3. Detects prominent spectral peaks
4. Generates constellation-map hashes
"""

import librosa
import numpy as np
from scipy.ndimage import maximum_filter


def load_audio(file_path, sr=22050):
    """Load an audio file as mono."""
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    return audio, sample_rate


def compute_spectrogram(audio, n_fft=4096, hop_length=512, window="hann"):
    """Return magnitude spectrogram in dB."""
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window
    )
    magnitude = np.abs(stft)
    return librosa.amplitude_to_db(magnitude, ref=np.max)


def detect_peaks(spectrogram, neighborhood_size=20, threshold_db=-45):
    """Detect strong local maxima."""
    local_max = maximum_filter(spectrogram, size=neighborhood_size)
    mask = (spectrogram == local_max) & (spectrogram > threshold_db)

    freq_bins, time_bins = np.where(mask)
    return list(zip(freq_bins, time_bins))


def generate_hashes(
    peaks,
    fan_value=10,
    min_time_delta=1,
    max_time_delta=200
):
    """Generate fingerprint hashes."""
    hashes = []

    peaks = sorted(peaks, key=lambda p: p[1])

    for i in range(len(peaks)):
        f1, t1 = peaks[i]

        for j in range(1, fan_value + 1):
            if i + j >= len(peaks):
                break

            f2, t2 = peaks[i + j]
            dt = t2 - t1

            if dt < min_time_delta or dt > max_time_delta:
                continue

            hashes.append(((int(f1), int(f2), int(dt)), int(t1)))

    return hashes


def fingerprint_song(file_path):
    """
    Complete fingerprint pipeline.

    Returns:
        dict containing spectrogram, peaks, hashes and sample rate.
    """
    audio, sr = load_audio(file_path)

    spectrogram = compute_spectrogram(audio)
    peaks = detect_peaks(spectrogram)
    hashes = generate_hashes(peaks)

    return {
        "spectrogram": spectrogram,
        "peaks": peaks,
        "hashes": hashes,
        "sample_rate": sr
    }
