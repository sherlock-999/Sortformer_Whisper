#!/usr/bin/env python3
"""
Sortformer_Whisper_Streaming: true streaming diarization + transcription pipeline.

For each audio file:
  1. Audio is split into fixed 480ms chunks (Sortformer's cadence).
  2. Each chunk is passed to StreamingSortformer, which returns a binary
     diarization mask [n_speakers, new_frames] for that chunk.
     Speaker identity is kept consistent across chunks internally.
  3. Mask chunks are accumulated alongside audio samples.
  4. Every `transcription_interval_s` seconds the accumulated audio and mask
     are passed to DiCoW_Pipeline for transcription.
  5. Segments are written to a JSONL file.

Usage:
    python Sortformer_Whisper_Streaming.py audio1.wav [audio2.wav ...]
    python Sortformer_Whisper_Streaming.py audio.wav --config config.yaml --interval 10.0
"""

import os
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import librosa
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.vad_utils import (
    load_postprocessing_from_yaml,
    predlist_to_timestamps,
)
from nemo.collections.asr.parts.utils.speaker_utils import timestamps_to_pyannote_object

from Sortformer.streaming_sortformer import StreamingSortformer
from dicow_inference import DiCoWTranscriber


SAMPLING_RATE = 16000


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Sortformer_Whisper_Streaming_Pipeline:

    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        self.cfg = OmegaConf.load(config_path)
        print(f"✓ Loaded config from {config_path}")

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ Device: {device_str}")

        print("\nLoading models...")
        self.sortformer  = StreamingSortformer(self.cfg.sortformer)
        self.transcriber = DiCoWTranscriber(self.cfg.dicow.model_path, device_str)
        self.postprocessing_cfg = load_postprocessing_from_yaml(
            self.cfg.sortformer.postprocessing_yaml
        )
        print("✓ Models loaded.\n")

    # ------------------------------------------------------------------
    def _preds_to_dicow_mask(
        self,
        accumulated_preds: torch.Tensor,
        audio_duration_s: float,
        session_id: str,
    ) -> torch.Tensor:
        """
        Convert accumulated raw Sortformer sigmoid predictions to a 50fps binary mask
        suitable for DiCoW, using the same postprocessing pipeline as the offline
        get_diarisation_mask.py (predlist_to_timestamps → timestamps_to_pyannote_object).

        Args:
            accumulated_preds : [n_spk, n_frames] raw sigmoid values at 12.5fps
            audio_duration_s  : wall-clock duration of the accumulated window in seconds
            session_id        : unique ID for this window (used by pyannote internals)

        Returns:
            50fps binary mask tensor of shape [n_spk, n_frames_50fps]
        """
        # predlist_to_timestamps expects [(1, n_frames, n_spk), ...]
        # unit_10ms_frame_count=8 → each frame = 8×10ms = 80ms = 12.5fps
        preds_for_pp = accumulated_preds.T.unsqueeze(0)  # [1, n_frames, n_spk]

        audio_rttm_map = {
            session_id: {
                "audio_filepath": session_id,
                "offset":         0.0,
                "duration":       audio_duration_s,
                "rttm_filepath":  None,
            }
        }

        cfg_vad = OmegaConf.structured(self.postprocessing_cfg)
        speaker_timestamps_list = predlist_to_timestamps(
            batch_preds_list=[preds_for_pp],
            audio_rttm_map_dict=audio_rttm_map,
            cfg_vad_params=cfg_vad,
            unit_10ms_frame_count=8,
            bypass_postprocessing=False,
        )
        # speaker_timestamps_list[0] is a list of per-speaker segment lists
        speaker_timestamps = speaker_timestamps_list[0]

        all_hypothesis = []
        timestamps_to_pyannote_object(
            speaker_timestamps=speaker_timestamps,
            uniq_id=session_id,
            audio_rttm_values=audio_rttm_map[session_id],
            all_hypothesis=all_hypothesis,
            all_reference=[],
            all_uems=[],
            out_rttm_dir=None,
        )

        # Convert pyannote annotation → 50fps binary mask
        # (same logic as get_diarisation_mask.py SortformerDiarizer.get_masks)
        annotation = all_hypothesis[0][1]  # pyannote Annotation object
        n_frames_50fps = round(audio_duration_s * 50)
        n_spk = accumulated_preds.shape[0]
        dicow_mask = torch.zeros(n_spk, n_frames_50fps)

        for spk_idx, spk_label in enumerate(annotation.labels()):
            for seg in annotation.label_timeline(spk_label):
                start_f = max(0, round(seg.start * 50))
                end_f   = min(n_frames_50fps, round(seg.end * 50))
                dicow_mask[spk_idx, start_f:end_f] = 1.0

        return dicow_mask

    # ------------------------------------------------------------------
    def process_audio_file(
        self,
        audio_path: str,
        transcription_interval_s: float = None,
    ) -> List[dict]:
        """
        Stream one audio file through Sortformer → DiCoW.

        Args:
            audio_path               : path to 16kHz mono WAV
            transcription_interval_s : override config value if provided

        Returns:
            list of output dicts (session_id, speaker, start_time, end_time, words)
        """
        interval_s = transcription_interval_s or self.cfg.pipeline.transcription_interval_s
        session_id = os.path.basename(audio_path).replace(".wav", "")

        print(f"\n{'='*60}")
        print(f"Processing : {session_id}")
        print(f"Interval   : {interval_s}s")
        print(f"{'='*60}")

        audio_full, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True, dtype=np.float32)
        duration = len(audio_full) / SAMPLING_RATE
        print(f"Duration   : {duration:.2f}s")

        chunk_samples    = self.sortformer.chunk_samples
        interval_samples = int(interval_s * SAMPLING_RATE)

        print("chunk_samples:", chunk_samples)
        print("interval_samples:", interval_samples)
        print("interval_s:", interval_s)

        # Reset Sortformer state — speaker cache cleared for new file
        self.sortformer.reset()

        accumulated_audio = np.zeros(0, dtype=np.float32)
        accumulated_mask  = None   # [n_spk, T_frames] float32 CPU tensor
        window_start_s    = 0.0

        all_results = []
        pos = 0

        while pos < len(audio_full):
            # ---- slice one chunk of true audio ----
            end        = min(pos + chunk_samples, len(audio_full))
            true_audio = audio_full[pos:end]

            # Pad to full chunk_samples so the convolutional pre-encoder sees a
            # complete receptive field, but pass true_length so the model's internal
            # length tracking matches streaming_feat_loader behaviour.
            true_length = len(true_audio)
            audio_chunk = true_audio
            if true_length < chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, chunk_samples - true_length))

            # ---- 1. Sortformer: one streaming step ----
            # Returns [n_spk, new_frames] raw sigmoid predictions for this chunk.
            # Speaker cache updated internally — no external state management needed.
            mask_chunk = self.sortformer.process_chunk(audio_chunk, true_length=true_length)

            # Accumulate true (un-padded) audio and mask
            accumulated_audio = np.concatenate([accumulated_audio, true_audio])
            accumulated_mask  = (
                mask_chunk if accumulated_mask is None
                else torch.cat([accumulated_mask, mask_chunk], dim=1)
            )

            pos = end

            # ---- 2. Transcribe when interval is full or audio exhausted ----
            if len(accumulated_audio) >= interval_samples or pos >= len(audio_full):
                window_end_s      = window_start_s + len(accumulated_audio) / SAMPLING_RATE
                window_duration_s = len(accumulated_audio) / SAMPLING_RATE

                # Convert raw Sortformer sigmoid preds → 50fps DiCoW mask via the
                # same postprocessing pipeline used in offline get_diarisation_mask.py
                dicow_mask = self._preds_to_dicow_mask(
                    accumulated_mask, window_duration_s, session_id
                )

                print(f"  → Transcribing {window_start_s:.1f}s – {window_end_s:.1f}s")

                for seg in self.transcriber.transcribe_with_masks(accumulated_audio, dicow_mask):
                    entry = {
                        "session_id": session_id,
                        "speaker":    seg["speaker"],
                        "start_time": round(window_start_s + seg["start_time"], 3),
                        "end_time":   round(window_start_s + seg["end_time"],   3),
                        "words":      seg["words"],
                    }
                    all_results.append(entry)
                    print(f"     [{entry['speaker']}] "
                          f"{entry['start_time']:.2f}s – {entry['end_time']:.2f}s : {entry['words']}")

                # Advance window
                window_start_s    = window_end_s
                accumulated_audio = np.zeros(0, dtype=np.float32)
                accumulated_mask  = None

        print(f"  ✓ {len(all_results)} segments.")
        return all_results

    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        mixed_list_path: str = None,
        output_dir: str = None,
        output_filename: str = None,
        transcription_interval_s: float = None,
    ):
        """
        Run streaming pipeline on a list of audio files and write output JSONL.

        Args:
            mixed_list_path: path to a text file with one audio path per line
                             (same format as Sortformer_Whisper.run_pipeline).
            output_dir:      override config output directory.
            output_filename: override config output filename.
            transcription_interval_s: override config transcription interval.

        Returns:
            List[dict]: all output segments (session_id, speaker, start_time, end_time, words).
        """
        if mixed_list_path is None:
            mixed_list_path = self.cfg.pipeline.get("mixed_list_path", None)
        if mixed_list_path is None:
            raise ValueError("mixed_list_path must be provided")

        with open(mixed_list_path) as f:
            audio_paths = [line.strip() for line in f if line.strip()]

        out_dir  = output_dir      or self.cfg.pipeline.output_dir
        out_file = output_filename or self.cfg.pipeline.output_filename
        os.makedirs(out_dir, exist_ok=True)
        output_path = Path(out_dir) / out_file

        all_results = []
        for audio_path in audio_paths:
            segments = self.process_audio_file(
                audio_path               = audio_path,
                transcription_interval_s = transcription_interval_s,
            )
            all_results.extend(segments)

        with open(output_path, "w", encoding="utf-8") as f:
            for entry in all_results:
                f.write(json.dumps(entry) + "\n")

        print(f"\n{'='*60}")
        print(f"✓ Saved {len(all_results)} segments → {output_path}")
        print(f"{'='*60}")
        return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="True streaming Sortformer + DiCoW pipeline"
    )
    parser.add_argument("mixed_list_path",
                        help="Path to text file with one audio path per line")
    parser.add_argument("--config",          default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--output_dir",      default=None,
                        help="Override output directory from config")
    parser.add_argument("--output_filename", default=None,
                        help="Override output filename from config")
    parser.add_argument("--interval",        type=float, default=None,
                        help="Override transcription_interval_s from config")
    args = parser.parse_args()

    pipeline = Sortformer_Whisper_Streaming_Pipeline(config_path=args.config)
    pipeline.run_pipeline(
        audio_paths              = args.audio_paths,
        output_dir               = args.output_dir,
        output_filename          = args.output_filename,
        transcription_interval_s = args.interval,
    )


if __name__ == "__main__":
    main()
