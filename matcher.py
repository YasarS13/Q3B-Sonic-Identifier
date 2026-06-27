from collections import defaultdict
import sqlite3
from fingerprint_b import AudioSignalProcessor
from database import DB_FILE_PATH, RelationalStorageEngine

class SubstringAudioMatcher:
    def __init__(self, db_path=DB_FILE_PATH):
        self.storage = RelationalStorageEngine(db_path)
        self.processor = AudioSignalProcessor()
        # REMOVE self.track_map from here

    def execute_identification(self, query_file_path):
        """Matches a sample file against the indexed relational signatures."""
        # 1. Fetch data safely dynamically
        track_map = self.storage.fetch_track_mapping() 
        
        audio, sr = self.processor.ingest_track(query_file_path)
        spec = self.processor.extract_density_features(audio)
        peaks = self.processor.isolate_salient_points(spec)
        query_fingerprints = self.processor.encode_triplet_hashes(peaks)

        # 2. Safety guard check
        if not query_fingerprints or not track_map:
            return self._build_empty_response(spec, peaks)

        # Structure: structural_offsets[track_id][time_difference_delta] -> match_counts
        structural_offsets = defaultdict(lambda: defaultdict(int))
        total_query_hashes = len(query_fingerprints)

        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()

        for q_fingerprint in query_fingerprints:
            q_hash = q_fingerprint["hash_id"]
            q_time = q_fingerprint["anchor_time"]

            cursor.execute("SELECT track_id, anchor_offset FROM signature_indices WHERE signature_hash = ?", (q_hash,))
            matches = cursor.fetchall()

            for track_id, db_time in matches:
                offset_delta = db_time - q_time
                structural_offsets[track_id][offset_delta] += 1

        conn.close()

        # 3. Explicitly initialize tracking states
        best_track_id = None
        best_score = 0
        target_track_histogram = {}

        # Locate the track with the single highest peak alignment bin
        for track_id, offsets_dict in structural_offsets.items():
            for offset, score in offsets_dict.items():
                if score > best_score:
                    best_score = score
                    best_track_id = track_id
                    target_track_histogram = offsets_dict

        # 4. Determine matching output metrics
        if best_track_id is not None and best_track_id in track_map:
            prediction = track_map[best_track_id]
            confidence = round((best_score / total_query_hashes) * 100, 2)
        else:
            prediction = "No Song Match"
            confidence = 0.0

        # 5. Return the payload response (Fixes the TypeError)
        return {
            "prediction": prediction,
            "confidence": min(confidence, 100.0),
            "votes": best_score,
            "spectrogram": spec,
            "peaks": peaks,
            "histogram_data": dict(target_track_histogram)
        }
