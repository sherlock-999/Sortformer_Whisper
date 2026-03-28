#!/usr/bin/env python3
"""
Sortformer_Whisper: Orchestrator for 3-step pipeline
1. Generate manifest from mixed_list.txt
2. Get diarization masks using Sortformer (in memory, no .pt files)
3. Run DiCoW transcription using masks
"""

import os
import sys
import json
import torch
import configparser
from pathlib import Path
from typing import Dict, Optional


from Sortformer.generate_manifest import ManifestGenerator
from Sortformer.get_diarisation_mask import SortformerDiarizer, DiarizationConfig
from dicow_inference import DiCoWTranscriber
from omegaconf import OmegaConf


class Sortformer_Whisper_Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: str = "config.ini"):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to config.ini file
        """
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.config.read(config_path)
        print(f"✓ Loaded config from {config_path}")
        
        # Initialize DiCoW transcriber
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dicow_model_path = self.config.get('DEFAULT', 'DICOW_MODEL_PATH')
        self.transcriber = DiCoWTranscriber(dicow_model_path, device)
        print(f"✓ DiCoW transcriber initialized on {device}")
    


    def step1_generate_manifest(self, mixed_list_path: str, manifest_path: str):
        """
        STEP 1: Generate manifest from mixed audio list.
        
        Args:
            mixed_list_path: Path to mixed_list.txt
            manifest_path: Output path for dataset_manifest.json
        """
        print("\n" + "="*60)
        print("STEP 1: Generating Manifest")
        print("="*60)
        print(f"Input:  {mixed_list_path}")
        print(f"Output: {manifest_path}")
        
        generator = ManifestGenerator()
        generator.generate(
            mixed_path_file=mixed_list_path,
            manifest_filepath=manifest_path,
            add_duration=True
        )
        
        print(f"✓ Manifest created: {manifest_path}")
        return manifest_path
    



    def step2_run_diarization(self, manifest_path: str) -> Dict[str, torch.Tensor]:
        """
        STEP 2: Run Sortformer diarization, get masks in memory.
        
        Args:
            manifest_path: Path to dataset_manifest.json
            
        Returns:
            Dict[audio_name] = speaker_mask (torch.Tensor)
        """
        print("\n" + "="*60)
        print("STEP 2: Running Sortformer Diarization")
        print("="*60)
        print(f"Input: {manifest_path}")
        
        # Create config from ini
        cfg = DiarizationConfig()
        cfg.model_path = self.config.get('DEFAULT', 'SORTFORMER_MODEL_PATH')
        cfg.dataset_manifest = manifest_path
        cfg.batch_size = 1
        cfg.num_workers = 0
        cfg.precision = "32"
        cfg.postprocessing_yaml = self.config.get('DEFAULT', 'POSTPROCESSING_YAML')
        cfg.bypass_postprocessing = self.config.getboolean('DEFAULT', 'BYPASS_POSTPROCESSING')
        cfg.no_der = True
        cfg.log = False
        
        # Set streaming params
        cfg.async_streaming = self.config.getboolean('DEFAULT', 'STREAMING')
        cfg.chunk_len = self.config.getint('DEFAULT', 'CHUNK_SIZE')
        cfg.chunk_right_context = self.config.getint('DEFAULT', 'RIGHT_CONTEXT')
        cfg.fifo_len = self.config.getint('DEFAULT', 'FIFO_SIZE')
        cfg.spkcache_update_period = self.config.getint('DEFAULT', 'UPDATE_PERIOD')
        cfg.spkcache_len = self.config.getint('DEFAULT', 'SPEAKER_CACHE_SIZE')
        cfg.async_streaming = True
        print(f"Streaming: ON")
        print(f"  CHUNK_SIZE={cfg.chunk_len}")
        print(f"  RIGHT_CONTEXT={cfg.chunk_right_context}")
        print(f"  FIFO_SIZE={cfg.fifo_len}")
        
        cfg = OmegaConf.structured(cfg)
        
        # Run diarization
        diarizer = SortformerDiarizer(cfg)
        masks = diarizer.get_masks()
        
        print(f"✓ Diarization complete: {len(masks)} audio files processed")
        print(f"✓ All masks in memory (no .pt files saved)")
        return masks
    



    def step3_run_transcription(self, manifest_path: str, masks: Dict[str, torch.Tensor], output_dir: str, output_filename: str = "hypothesis_multi.jsonl"):
        """
        STEP 3: Run DiCoW transcription with masks (in-memory, no .pt files).
        
        Args:
            manifest_path: Path to dataset_manifest.json
            masks: Dict[audio_name] = mask from step 2
            output_dir: Output directory for transcriptions
            output_filename: Name for the output JSONL file (default: "hypothesis_multi.jsonl")
        """
        print("\n" + "="*60)
        print("STEP 3: Running DiCoW Transcription")
        print("="*60)
        print(f"Masks: {len(masks)} audio files (in memory)")
        print(f"Output: {output_dir}")
        print(f"Output filename: {output_filename}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Pass masks directly to transcriber - no .pt files needed
        self.transcriber.transcribe_with_masks(manifest_path, masks, output_dir, output_filename)
        
        print(f"✓ Transcriptions saved to {output_dir}")
        print(f"✓ All masks processed directly from memory (no temp files)")
    

    
    def run_pipeline(self, mixed_list_path: str = None, output_dir: str = None, output_filename: str = "hypothesis_multi.jsonl"):
        """
        Run complete 3-step pipeline.
        
        Args:
            mixed_list_path: Path to mixed_list.txt (uses config if None)
            output_dir: Output directory (uses config if None)
            output_filename: Name for the output JSONL file (default: "hypothesis_multi.jsonl")
        """
        if mixed_list_path is None:
            mixed_list_path = self.config.get('DEFAULT', 'MIXED_LIST_PATH')
        
        if output_dir is None:
            output_dir = self.config.get('DEFAULT', 'OUTPUT_TRANSCRIPTION_DIR')
        
        # Generate manifest filename from output_filename (e.g., "nsf_hypothesis_multi.jsonl" -> "nsf_hypothesis_multi_manifest.json")
        manifest_filename = output_filename.replace('.jsonl', '_manifest.json')
        manifest_dir = self.config.get('DEFAULT', 'OUTPUT_TRANSCRIPTION_DIR')
        manifest_path = os.path.join(manifest_dir, manifest_filename)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(manifest_dir, exist_ok=True)
        
        # Step 1
        self.step1_generate_manifest(mixed_list_path, manifest_path)
        
        # Step 2
        masks = self.step2_run_diarization(manifest_path)
        
        # Step 3
        self.step3_run_transcription(manifest_path, masks, output_dir, output_filename)

        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETE")
        print("="*60)

        return masks


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sortformer_Whisper Pipeline")
    parser.add_argument("--config", default="Sortformer/config.ini")
    parser.add_argument("--mixed_list", help="Path to mixed_list.txt")
    parser.add_argument("--output_dir", help="Output directory for transcriptions")
    parser.add_argument("--output_filename", default="hypothesis_multi.jsonl", help="Output JSONL filename (default: hypothesis_multi.jsonl)")
    
    args = parser.parse_args()
    
    pipeline = Sortformer_Whisper_Pipeline(args.config)
    pipeline.run_pipeline(args.mixed_list, args.output_dir, args.output_filename)


if __name__ == "__main__":
    main()
