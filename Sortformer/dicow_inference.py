import os
import sys
import torch
from pathlib import Path
from typing import Dict, Optional

# Add model directory to path for DiCoW imports
model_dir = str(Path(__file__).parent.parent / "model")
print(f"Adding model directory to path: {model_dir}")
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

from transformers import AutoTokenizer, AutoFeatureExtractor

from DiCoW.modeling_dicow import DiCoWForConditionalGeneration
from Sortformer.dicow_pipeline import DiCoW_Pipeline

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
    


    def transcribe_with_masks(self, manifest_path: str, masks: Dict[str, torch.Tensor], output_dir: str):
        """
        Transcribe audio files using diarization masks for all speakers.
        
        Args:
            manifest_path: Path to dataset_manifest.json
            masks: Dict[audio_name] = mask (torch.Tensor with shape [num_speakers, num_frames])
            output_dir: Output directory for transcriptions
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load manifest
        with open(manifest_path, "r") as f:
            manifest_items = [json.loads(line) for line in f.readlines()]
        
        print(f"Processing {len(manifest_items)} audio files...")
        
        # Process each audio file
        for item in manifest_items:
            mixed_audio_path = item["audio_filepath"]
            mixed_audio_name = os.path.basename(mixed_audio_path).replace(".wav", "")
            
            # Get mask from dictionary
            if mixed_audio_name not in masks:
                print(f"⚠ Warning: No mask found for {mixed_audio_name}, skipping")
                continue
            
            diarization_mask = masks[mixed_audio_name]
            num_speakers = diarization_mask.shape[0]
            
            # Set mask on pipeline
            self.pipeline.diarization_mask = diarization_mask
            
            # Process each speaker
            for speaker_id in range(num_speakers):
                print(f"\n  Processing Speaker {speaker_id}...")
                
                # Run inference for this speaker
                inputs = {
                    "audio_filepath": mixed_audio_path,
                    "target_speaker_id": speaker_id
                }
                result = self.pipeline(inputs, return_timestamps=True)
                
                # Get transcription for this speaker (pipeline returns list with 1 element per speaker)
                speaker_transcription = result["per_spk_outputs"][0]
                
                # Save transcription with speaker ID
                out_path = Path(output_dir) / f"{mixed_audio_name}_speaker{speaker_id}.txt"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(speaker_transcription)
                
                print(f"  ✓ Saved Speaker {speaker_id}: {out_path}")
            
            # Reset mask for next iteration
            self.pipeline.diarization_mask = None


def main():
    parser = argparse.ArgumentParser(description="DiCoW batch inference from manifest")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSONL")
    parser.add_argument("--diar_mask_dir", type=str, required=True, help="Directory containing diarization masks (.pt)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save transcriptions")
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