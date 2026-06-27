import hashlib
import numpy as np
import librosa
from scipy.ndimage import maximum_filter

class AudioSignalProcessor:
    def __init__(self, sample_rate=9600):
        self.sample_rate = sample_rate

    def ingest_track(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio, sr

    def extract_density_features(self, audio, n_fft=4096, hop_length=512):
        stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window="hann")
        magnitude = np.abs(stft_matrix)
        db_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
        return db_spectrogram.astype(np.float32)

    def isolate_salient_points(self, spectrogram, size=20, threshold=-65):
        local_maxima = maximum_filter(spectrogram, size=size)
        matching_mask = (spectrogram == local_maxima) & (spectrogram > threshold)
        
        freq_coords, time_coords = np.where(matching_mask)
        
        peaks = []
        for f, t in zip(freq_coords, time_coords):
            peaks.append({"frequency": int(f), "time_frame": int(t)})
        
        peaks.sort(key=lambda x: x["time_frame"])
        return peaks

    def encode_triplet_hashes(self, peaks, stride=1):
        fingerprints = []
        num_peaks = len(peaks)
        
        if num_peaks < 3:
            print("[DEBUG] Not enough peaks to construct a single triplet.")
            return fingerprints

        for i in range(0, num_peaks - 2, stride):
            p1 = peaks[i]
            p2 = peaks[i + 1]
            p3 = peaks[i + 2]
            
            t_delta_1 = p2["time_frame"] - p1["time_frame"]
            t_delta_2 = p3["time_frame"] - p2["time_frame"]
            
            # WIDENED WINDOW FOR SHORTER CLIPS: Changed lower bound from 1 to 0 
            # to accommodate tightly packed frames
            if (0 <= t_delta_1 <= 200) and (0 <= t_delta_2 <= 200):
                token_string = f"{p1['frequency']}|{p2['frequency']}|{p3['frequency']}:{t_delta_1}:{t_delta_2}"
                hex_digest = hashlib.md5(token_string.encode('utf-8')).hexdigest()
                hash_id = int(hex_digest[:8], 16)
                
                fingerprints.append({
                    "hash_id": hash_id,
                    "anchor_time": p1["time_frame"]
                })
                
        print(f"[DEBUG] Successfully encoded {len(fingerprints)} triplet hashes.")
        return fingerprints