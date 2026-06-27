import os
import sqlite3
import numpy as np
from fingerprint import AudioSignalProcessor

DB_FILE_PATH = "fingerprints_relational.db"

class RelationalStorageEngine:
    def __init__(self, db_path=DB_FILE_PATH):
        self.db_path = db_path
        self._initialize_tables()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Track track entries explicitly
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS registry_tracks (
                    track_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_name TEXT UNIQUE
                )
            """)
            # Store fingerprints mapped back to registry identifiers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signature_indices (
                    signature_hash INTEGER,
                    track_id INTEGER,
                    anchor_offset INTEGER,
                    FOREIGN KEY(track_id) REFERENCES registry_tracks(track_id)
                )
            """)
            # Create indexing structures to handle lightning fast query responses
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sig_hash ON signature_indices(signature_hash)")
            conn.commit()

    def register_new_track(self, track_name):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO registry_tracks (track_name) VALUES (?)", (track_name,))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                cursor.execute("SELECT track_id FROM registry_tracks WHERE track_name = ?", (track_name,))
                return cursor.fetchone()[0]

    def insert_fingerprints(self, track_id, fingerprints):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            records = [(f["hash_id"], track_id, f["anchor_time"]) for f in fingerprints]
            cursor.executemany("INSERT INTO signature_indices VALUES (?, ?, ?)", records)
            conn.commit()

    def fetch_track_mapping(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT track_id, track_name FROM registry_tracks")
            return {row[0]: row[1] for row in cursor.fetchall()}


def build_database(song_folder="songs"):
    if not os.path.isdir(song_folder):
        raise FileNotFoundError(f"Target directory {song_folder} does not exist.")
        
    supported_extensions = (".mp3", ".wav", ".flac")
    files = sorted(f for f in os.listdir(song_folder) if f.lower().endswith(supported_extensions))
    
    if not files:
        raise RuntimeError("No compatible audio clips discovered.")

    processor = AudioSignalProcessor()
    storage = RelationalStorageEngine()

    print(f"Initializing batch ingestion sequence for {len(files)} items...\n")
    
    for filename in files:
        full_path = os.path.join(song_folder, filename)
        track_name = os.path.splitext(filename)[0]
        
        print(f" -> Mapping features for target: {track_name}")
        
        try:
            audio, sr = processor.ingest_track(full_path)
            spec = processor.extract_density_features(audio)
            peaks = processor.isolate_salient_points(spec)
            fingerprints = processor.encode_triplet_hashes(peaks)
            
            track_id = storage.register_new_track(track_name)
            storage.insert_fingerprints(track_id, fingerprints)
        except Exception as err:
            print(f" [WARNING] Failed to catalog {track_name}: {err}")

    print("\nRelational SQL database build complete.")
