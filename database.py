"""
database.py

Indexes every song inside the songs/ directory and builds a
fingerprint database for fast lookup.
"""

import os
import pickle
from collections import defaultdict
from fingerprint import fingerprint_song


SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")


def build_database(song_folder="songs", output_file="database/fingerprints.pkl"):
    """
    Fingerprint every song and save the database.

    Database structure
    ------------------
    {
        "hash_map": {
            hash_key: [
                (song_name, anchor_time),
                ...
            ]
        },
        "songs": [
            song_name1,
            song_name2,
            ...
        ]
    }
    """

    if not os.path.isdir(song_folder):
        raise FileNotFoundError(f"Song folder not found: {song_folder}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    hash_map = defaultdict(list)
    songs = []

    files = sorted(
        f for f in os.listdir(song_folder)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )

    if not files:
        raise RuntimeError("No supported audio files found.")

    print(f"Found {len(files)} songs.\n")

    for index, filename in enumerate(files, start=1):
        full_path = os.path.join(song_folder, filename)
        song_name = os.path.splitext(filename)[0]

        print(f"[{index}/{len(files)}] Indexing: {song_name}")

        result = fingerprint_song(full_path)

        songs.append(song_name)

        for hash_key, anchor_time in result["hashes"]:
            hash_map[hash_key].append((song_name, anchor_time))

    database = {
        "hash_map": dict(hash_map),
        "songs": songs
    }

    with open(output_file, "wb") as f:
        pickle.dump(database, f)

    print("\nDatabase saved to:", output_file)
    print("Unique hashes:", len(database["hash_map"]))
    print("Songs indexed:", len(songs))

    return database


def load_database(database_file="database/fingerprints.pkl"):
    """Load an existing fingerprint database."""
    with open(database_file, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    build_database()
