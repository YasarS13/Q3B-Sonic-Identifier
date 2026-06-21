"""
matcher.py

Audio fingerprint matching engine.

Loads the fingerprint database, fingerprints a query clip,
matches hashes, computes offset histograms and predicts
the most likely song.
"""

import pickle
from collections import Counter, defaultdict

from fingerprint import fingerprint_song


class SongMatcher:
    def __init__(self, database_path="database/fingerprints.pkl"):
        with open(database_path, "rb") as f:
            self.database = pickle.load(f)

        self.hash_map = self.database["hash_map"]

    def match(self, query_file):
        """
        Match a query audio clip.

        Returns
        -------
        dict
        """

        result = fingerprint_song(query_file)

        query_hashes = result["hashes"]

        offset_votes = defaultdict(Counter)

        total_matches = 0

        for hash_key, query_time in query_hashes:

            if hash_key not in self.hash_map:
                continue

            matches = self.hash_map[hash_key]

            total_matches += len(matches)

            for song_name, db_time in matches:

                offset = db_time - query_time

                offset_votes[song_name][offset] += 1

        if len(offset_votes) == 0:

            return {
                "prediction": "No Match",
                "confidence": 0,
                "offset_histogram": {},
                "spectrogram": result["spectrogram"],
                "peaks": result["peaks"]
            }

        best_song = None
        best_offset = None
        best_score = -1

        histogram = {}

        for song, counter in offset_votes.items():

            offset, votes = counter.most_common(1)[0]

            histogram[song] = counter

            if votes > best_score:

                best_score = votes
                best_song = song
                best_offset = offset

        confidence = 0

        if total_matches > 0:
            confidence = round(
                100 * best_score / total_matches,
                2
            )

        return {
            "prediction": best_song,
            "confidence": confidence,
            "offset": best_offset,
            "votes": best_score,
            "offset_histogram": histogram,
            "spectrogram": result["spectrogram"],
            "peaks": result["peaks"]
        }


if __name__ == "__main__":

    matcher = SongMatcher()

    query = input("Query audio path: ")

    result = matcher.match(query)

    print("\nPrediction :", result["prediction"])
    print("Confidence:", result["confidence"], "%")
    print("Votes:", result["votes"])
