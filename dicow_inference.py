import os
import sys
import torch
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

from transformers import AutoTokenizer, AutoFeatureExtractor

from model.DiCoW.modeling_dicow import DiCoWForConditionalGeneration
from dicow_pipeline import DiCoW_Pipeline

import argparse
import json

# -----------------------------
# Helper
# -----------------------------
def create_lower_uppercase_mapping(tokenizer):
    tokenizer.upper_cased_tokens = {}
    vocab = tokenizer.get_vocab()
    for token, index in vocab.items():
        if len(token) < 1:
            continue
        if token[0] == 'Ġ' and len(token) > 1:
            lower = token[0] + token[1].lower() + (token[2:] if len(token) > 2 else '')
        else:
            lower = token[0].lower() + token[1:]
        if lower != token and lower in vocab:
            tokenizer.upper_cased_tokens[vocab[lower]] = index



class DiCoWTranscriber:
    """DiCoW transcriber that accepts masks directly from memory."""
    
    def __init__(self, dicow_model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize DiCoW transcriber.
        
        Args:
            dicow_model_path: Path to DiCoW model directory (uses default if None)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        if dicow_model_path is None:
            dicow_model_path = DICOW_MODEL_PATH
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Initializing DiCoW on {self.device}...")
        
        # Load model
        self.dicow = DiCoWForConditionalGeneration.from_pretrained(
            dicow_model_path,
            local_files_only=True
        ).to(self.device)
        
        # Load tokenizer and feature extractor
        self.tokenizer = AutoTokenizer.from_pretrained(
            dicow_model_path,
            local_files_only=True
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            dicow_model_path,
            local_files_only=True
        )
        
        # Setup tokenizer
        create_lower_uppercase_mapping(self.tokenizer)
        self.dicow.set_tokenizer(self.tokenizer)
        
        # Initialize pipeline
        self.pipeline = DiCoW_Pipeline(
            self.dicow,
            speaker_embedding_model=None,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        print("✓ DiCoW initialized successfully")
    


    def transcribe_with_masks(
        self,
        audio,
        diarization_mask: torch.Tensor,
    ) -> List[dict]:
        """
        Transcribe a single audio with its diarization mask.

        Args:
            audio:             file path (str) or float32 numpy/torch array at 16kHz.
            diarization_mask:  tensor of shape [num_speakers, num_frames] at 50 fps.

        Returns:
            list of segment dicts (speaker, start_time, end_time, words).
        """
        if isinstance(audio, str):
            pipeline_input = {"audio_filepath": audio, "diarization_mask": diarization_mask}
        else:
            pipeline_input = {"audio": audio, "diarization_mask": diarization_mask}

        result = self.pipeline(pipeline_input, return_timestamps=True)

        segments = []
        for spk_idx, speaker_transcription in enumerate(result.get("per_spk_outputs", [])):
            if not speaker_transcription or not speaker_transcription.strip():
                continue
            for start_time, end_time, text in self._extract_segments_with_timing(speaker_transcription):
                if text.strip():
                    segments.append({
                        "speaker":    f"speaker_{spk_idx}",
                        "start_time": start_time,
                        "end_time":   end_time,
                        "words":      text,
                    })
        return segments
    
    @staticmethod
    def _extract_segments_with_timing(processed_text: str) -> List[Tuple[float, float, str]]:
        """
        Extract segments with timing from processed text.
        Example: '<|237.28|>text1<|245.92|><|246.24|>text2<|250.0|>'
        Returns: [(237.28, 245.92, 'text1'), (246.24, 250.0, 'text2')]
        """
        import re
        
        # Find segments with timing pairs: <|start|>...text...<|end|>
        segments = []
        pattern = r'<\|([\d.]+)\|>'
        
        # Find all timestamps and their positions
        matches = list(re.finditer(pattern, processed_text))
        
        # Group timestamps in pairs (start, end) and extract text between them
        for i in range(0, len(matches) - 1, 2):
            start_match = matches[i]
            end_match = matches[i + 1]
            
            start_time = float(start_match.group(1))
            end_time = float(end_match.group(1))
            
            # Extract text between the two timestamps
            text_start = start_match.end()
            text_end = end_match.start()
            text = processed_text[text_start:text_end].strip()
            
            if text:  # Only add non-empty segments
                segments.append((start_time, end_time, text))
        
        return segments if segments else [(0.0, 0.0, processed_text)]


def main():
    parser = argparse.ArgumentParser(description="DiCoW batch inference from manifest")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSONL")
    parser.add_argument("--diar_mask_dir", type=str, required=True, help="Directory containing diarization masks (.pt)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save transcriptions")
    parser.add_argument("--output_filename", type=str, default="hypothesis_multi.jsonl", help="Output JSONL filename (default: hypothesis_multi.jsonl)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------
    # Load DiCoW model + tokenizer + feature extractor
    # -------------------------------
    dicow = DiCoWForConditionalGeneration.from_pretrained(
        DICOW_MODEL_PATH,
        local_files_only=True
    ).to(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        DICOW_MODEL_PATH,
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        DICOW_MODEL_PATH,
        local_files_only=True
    )

    create_lower_uppercase_mapping(tokenizer)
    dicow.set_tokenizer(tokenizer)

    # -------------------------------
    # Speaker verification model (optional)
    # -------------------------------
    speaker_verification_model = None  # Set to None if using precomputed diarization masks

    # -------------------------------
    # Initialize pipeline
    # -------------------------------
    pipeline = DiCoW_Pipeline(
        dicow,
        speaker_embedding_model=speaker_verification_model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device
    )

    # -------------------------------
    # Batch inference from manifest
    # -------------------------------
    with open(args.manifest, "r") as f:
        manifest_items = [json.loads(line) for line in f.readlines()]

    for item in manifest_items:

        mixed_audio_path = item["mixed_filepath"]

        # Load corresponding diarization mask
        mixed_audio_name = os.path.basename(mixed_audio_path).replace(".wav", "")
        diar_mask_path = os.path.join(args.diar_mask_dir, f"{mixed_audio_name}_mask.pt")
        diarization_mask = torch.load(diar_mask_path)
        pipeline.diarization_mask = diarization_mask

        # Run inference
        inputs = {
            "mixed_audio_path": mixed_audio_path
        }
        result = pipeline(inputs, return_timestamps=True)

        target_speaker_transcription = result["per_spk_outputs"][0]

        out_path = Path(args.output_dir) / f"{mixed_audio_name}_transcription.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(target_speaker_transcription)

        print(f"Saved transcription for {mixed_audio_name} -> {out_path}")


if __name__ == "__main__":
    main()