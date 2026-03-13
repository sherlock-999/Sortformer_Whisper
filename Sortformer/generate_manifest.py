# generate_manifest.py
# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0.

import json
import librosa
import random
from pathlib import Path

random.seed(42)


class ManifestGenerator:
    """Class to generate dataset manifest from mixed audio list."""
    
    def generate(self, mixed_path_file: str, manifest_filepath: str, add_duration: bool = False) -> str:
        """
        Generate manifest JSON file from mixed audio list.
        
        Args:
            mixed_path_file: Path to text file with audio file paths (one per line)
            manifest_filepath: Output path for manifest JSON file
            add_duration: Whether to compute and add audio duration
            
        Returns:
            Path to generated manifest file
        """
        mixed_path_file = Path(mixed_path_file)
        if not mixed_path_file.is_file():
            raise FileNotFoundError(f"Mixed audio list file not found: {mixed_path_file}")

        # Read mixed audio files
        with open(mixed_path_file, "r") as f:
            wav_files = [line.strip() for line in f.readlines() if line.strip()]
        if len(wav_files) == 0:
            raise ValueError("No mixed audio files found in the input file.")

        # Create manifest entries
        manifest_entries = []
        for i, wav in enumerate(wav_files):
            entry = {
                "audio_filepath": wav,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": None,
                "num_speakers": None,
                "rttm_filepath": None,
                "uem_filepath": None,
                "ctm_filepath": None
            }
            if add_duration:
                try:
                    y, sr = librosa.load(wav, sr=None)
                    entry["duration"] = float(len(y) / sr)
                except Exception as e:
                    print(f"Warning: Could not load {wav}: {e}")
            manifest_entries.append(entry)

        # Write manifest file
        manifest_path = Path(manifest_filepath)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry) + "\n")

        print(f"Manifest saved to {manifest_path}")
        return str(manifest_path)


# Backward compatibility: keep old function
def main(mixed_path_file, manifest_filepath=None, add_duration=False):
    generator = ManifestGenerator()
    generator.generate(mixed_path_file, manifest_filepath, add_duration)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths2mixed_files",
        help="Path to text file containing list of mixed audio files",
        type=str,
        required=True
    )
    parser.add_argument(
        "--manifest_filepath",
        help="Path to output manifest file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--add_duration",
        help="Add duration of audio files to output manifest",
        action='store_true'
    )

    args = parser.parse_args()

    generator = ManifestGenerator()
    generator.generate(
        mixed_path_file=args.paths2mixed_files,
        manifest_filepath=args.manifest_filepath,
        add_duration=args.add_duration
    )