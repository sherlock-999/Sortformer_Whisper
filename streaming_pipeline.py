#!/usr/bin/env python3
"""
True streaming pipeline: Sortformer chunk-by-chunk → accumulated diarization mask → DiCoW.

Architecture:
  - Outer loop feeds fixed 480ms audio chunks (= chunk_len * subsampling_factor frames)
  - Each chunk goes to:
      1. Streaming Sortformer (forward_streaming_step) → new mask frames appended to accumulated_mask
      2. DiCoWOnlineASRProcessor.insert_audio_chunk() → audio buffer grows
  - DiCoW is called by process_iter() whenever the audio buffer is large enough
  - Buffer trimming (chunk_at) trims BOTH audio buffer and accumulated_mask in sync
  - Single target speaker (speaker 0)
  - No VAD — Sortformer cadence drives the loop
"""

import sys
import math
import re
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import configparser
from pathlib import Path
from typing import Optional, Tuple

from transformers import AutoTokenizer, AutoFeatureExtractor

from model.DiCoW.modeling_dicow import DiCoWForConditionalGeneration
from dicow_inference import create_lower_uppercase_mapping
from whisper_streaming import HypothesisBuffer, OnlineASRProcessor

from nemo.collections.asr.models import SortformerEncLabelModel
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLING_RATE = 16000
# Sortformer encoder subsampling factor (8x) × chunk_len (6 frames) = 48 feature frames
# Feature frame stride = 10ms → 48 × 10ms = 480ms per chunk
SORTFORMER_SUBSAMPLING = 8
SORTFORMER_FEATURE_STRIDE_MS = 10          # ms per mel frame
# Sortformer output frame rate after subsampling: 1 frame per 80ms
MASK_FRAME_RATE = 1000 / (SORTFORMER_SUBSAMPLING * SORTFORMER_FEATURE_STRIDE_MS)  # 12.5 fps
TARGET_SPEAKER = 0                         # which speaker slot to transcribe


# ---------------------------------------------------------------------------
# DiCoW streaming ASR backend  (drop-in replacement for FasterWhisperASR)
# ---------------------------------------------------------------------------

class DiCoWStreamingASR:
    """
    Wraps DiCoWForConditionalGeneration for use inside OnlineASRProcessor.

    The stno_mask is injected externally (by DiCoWOnlineASRProcessor) before
    each call to transcribe().  This keeps the interface compatible with
    whisper_streaming's ASRBase contract.
    """

    sep = ""   # faster-whisper style: spaces already embedded in tokens

    def __init__(self, model, feature_extractor, tokenizer, device,
                 language="en", target_speaker=TARGET_SPEAKER):
        self.model = model
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.device = device
        self.original_language = language
        self.target_speaker = target_speaker

        # Set externally by DiCoWOnlineASRProcessor before each transcribe()
        self.stno_mask = None

    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        """
        audio     : float32 numpy array at 16 kHz (variable length)
        init_prompt: committed text used as Whisper prompt context

        Returns the raw generate() output (token sequences + timestamps).
        ts_words() parses it into [(start, end, word), ...].
        """
        # --- mel features (always padded to 30s by the feature extractor) ---
        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)  # [1, 80, 3000]

        # --- prompt ids ---
        prompt_ids = None
        if init_prompt:
            prompt_ids = self.tokenizer(
                init_prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(self.device).squeeze(0)

        generate_kwargs = dict(
            input_features=input_features,
            return_timestamps=True,
            return_token_timestamps=True,
            return_segments=True,
            language=self.original_language,
            task="transcribe",
        )
        if self.stno_mask is not None:
            generate_kwargs["stno_mask"] = self.stno_mask
        if prompt_ids is not None:
            generate_kwargs["prompt_ids"] = prompt_ids

        with torch.inference_mode():
            outputs = self.model.generate(**generate_kwargs)

        return outputs

    # ------------------------------------------------------------------
    def ts_words(self, outputs) -> list:
        """
        Parse DiCoW generate() outputs into [(start_s, end_s, segment_text), ...].

        DiCoW decodes with timestamps in the form "<|t0|>text<|t1|>".
        We treat each timestamp-bracketed segment as one unit (not word-level),
        which is what HypothesisBuffer needs: consecutive transcriptions must
        produce matching (start, end, text) tuples for commitment.
        """
        if outputs is None:
            return []

        # get token id sequences from generate() output dict
        if isinstance(outputs, dict):
            sequences = outputs.get("sequences", None)
        else:
            sequences = outputs

        if sequences is None:
            return []

        # Decode exactly as DiCoW_Pipeline.postprocess does
        decoded_texts = self.tokenizer.batch_decode(
            sequences,
            decode_with_timestamps=True,
            skip_special_tokens=True,
        )

        segments = []
        ts_pattern = re.compile(r"<\|([\d.]+)\|>")
        for text in decoded_texts:
            matches = list(ts_pattern.finditer(text))
            # timestamps come in start/end pairs: <|t0|>...text...<|t1|>
            for i in range(0, len(matches) - 1, 2):
                start_t  = float(matches[i].group(1))
                end_t    = float(matches[i + 1].group(1))
                seg_text = text[matches[i].end() : matches[i + 1].start()].strip()
                if seg_text:
                    segments.append((start_t, end_t, seg_text))

        return segments

    # ------------------------------------------------------------------
    def segments_end_ts(self, outputs) -> list:
        """
        Return end timestamps of all decoded segments.
        Used by OnlineASRProcessor.chunk_completed_segment() to decide trim point.
        """
        return [end for (_, end, _) in self.ts_words(outputs)]

    def use_vad(self):
        pass  # VAD disabled — Sortformer drives cadence


# ---------------------------------------------------------------------------
# DiCoW-aware OnlineASRProcessor: co-manages the diarization mask buffer
# ---------------------------------------------------------------------------

class DiCoWOnlineASRProcessor(OnlineASRProcessor):
    """
    Extends OnlineASRProcessor with an accumulated_mask buffer that mirrors
    audio_buffer in time.  chunk_at() trims both in sync.

    Before each process_iter() call, the STNO mask is built from
    accumulated_mask and injected into asr.stno_mask.
    """

    MASK_FRAME_RATE = MASK_FRAME_RATE  # 12.5 fps

    def __init__(self, asr: DiCoWStreamingASR, n_speakers: int = 4,
                 buffer_trimming=("segment", 15), logfile=sys.stderr):
        # No tokenizer needed (segment-based trimming default)
        super().__init__(asr, tokenizer=None,
                         buffer_trimming=buffer_trimming, logfile=logfile)
        self.n_speakers = n_speakers
        self.accumulated_mask = None   # [n_speakers, T_mask_frames], float32 on CPU

    # ------------------------------------------------------------------
    def init(self, offset=None):
        super().init(offset=offset)
        self.accumulated_mask = None

    # ------------------------------------------------------------------
    def insert_mask_chunk(self, mask_chunk: torch.Tensor):
        """
        Append new diarization frames to the accumulated mask.

        mask_chunk: [n_speakers, T_chunk_frames] float32 (0/1 binary or sigmoid)
        """
        mask_cpu = mask_chunk.cpu().float()
        if self.accumulated_mask is None:
            self.accumulated_mask = mask_cpu
        else:
            self.accumulated_mask = torch.cat([self.accumulated_mask, mask_cpu], dim=1)

    # ------------------------------------------------------------------
    def _build_stno_mask(self) -> Optional[torch.Tensor]:
        """
        Build a [1, 4, 1500] STNO mask for the target speaker from accumulated_mask,
        aligned to the current audio_buffer window, interpolated to 1500 encoder frames.

        STNO channels (per DiCoW convention):
          0: silence
          1: target speaker only
          2: non-target, no target
          3: overlap (target + anyone else)
        """
        if self.accumulated_mask is None:
            return None

        diar = self.accumulated_mask  # [n_spk, T]
        n_spk = diar.shape[0]
        if n_spk == 0:
            return None

        s = TARGET_SPEAKER
        non_target_mask = torch.ones(n_spk, dtype=torch.bool)
        non_target_mask[s] = False

        sil_frames       = (1.0 - diar).prod(dim=0)                         # silence
        anyone_else      = (1.0 - diar[non_target_mask]).prod(dim=0)         # no one but target
        target_only      = diar[s] * anyone_else                              # target, alone
        non_target_only  = (1.0 - diar[s]) * (1.0 - anyone_else)             # others, not target
        overlap          = diar[s] - target_only                              # target + others

        stno = torch.stack([sil_frames, target_only, non_target_only, overlap], dim=0)
        # stno: [4, T]  →  [1, 4, T]  →  interpolate  →  [1, 4, 1500]
        stno = stno.unsqueeze(0)
        stno = F.interpolate(stno.float(), size=1500, mode="nearest")

        return stno.to(self.asr.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    def chunk_at(self, time: float):
        """
        Override: trim both audio_buffer and accumulated_mask from the front
        by the same duration, keeping them perfectly aligned.
        """
        cut_seconds = time - self.buffer_time_offset
        cut_frames  = int(cut_seconds * self.MASK_FRAME_RATE)

        if self.accumulated_mask is not None and cut_frames > 0:
            self.accumulated_mask = self.accumulated_mask[:, cut_frames:]

        super().chunk_at(time)   # trims audio_buffer and updates buffer_time_offset

    # ------------------------------------------------------------------
    def process_iter(self):
        """
        Inject the current STNO mask into asr before transcribing.
        """
        self.asr.stno_mask = self._build_stno_mask()
        return super().process_iter()


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_sortformer(model_path: str, device: torch.device,
                    chunk_len: int = 6,
                    fifo_len: int = 188,
                    spkcache_len: int = 188,
                    spkcache_update_period: int = 144):
    """Load the Sortformer model and configure it for true streaming."""
    map_location = device
    diar_model = SortformerEncLabelModel.restore_from(
        restore_path=model_path,
        map_location=map_location
    )
    diar_model = diar_model.eval().to(device)

    # Must have streaming_mode=True in the .nemo config for forward_streaming to work
    if not diar_model.streaming_mode:
        raise RuntimeError(
            "Sortformer model was not saved with streaming_mode=True. "
            "Use the streaming variant of the .nemo checkpoint."
        )

    # Configure streaming parameters
    diar_model.async_streaming = True
    diar_model.sortformer_modules.chunk_len              = chunk_len
    diar_model.sortformer_modules.fifo_len               = fifo_len
    diar_model.sortformer_modules.spkcache_len           = spkcache_len
    diar_model.sortformer_modules.spkcache_update_period = spkcache_update_period
    diar_model.sortformer_modules.chunk_left_context     = 0   # no look-ahead: true online
    diar_model.sortformer_modules.chunk_right_context    = 0

    return diar_model


def load_dicow(model_path: str, device: torch.device):
    """Load DiCoW model, tokenizer, and feature extractor."""
    model = DiCoWForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)

    create_lower_uppercase_mapping(tokenizer)
    model.set_tokenizer(tokenizer)

    return model, tokenizer, feature_extractor


# ---------------------------------------------------------------------------
# Main streaming function
# ---------------------------------------------------------------------------

def run_streaming(
    audio_path: str,
    sortformer_model_path: str,
    dicow_model_path: str,
    language: str = "en",
    min_chunk_size_s: float = 1.0,       # DiCoW fires only when buffer >= this
    buffer_trim_sec: float = 15.0,       # trim audio buffer when longer than this
    chunk_len: int = 6,                  # Sortformer chunk_len in diar frames
    fifo_len: int = 188,
    spkcache_len: int = 188,
    spkcache_update_period: int = 144,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------------------------------------------------------------
    # Load models
    # ---------------------------------------------------------------
    print("Loading Sortformer...")
    diar_model = load_sortformer(
        sortformer_model_path, device,
        chunk_len=chunk_len,
        fifo_len=fifo_len,
        spkcache_len=spkcache_len,
        spkcache_update_period=spkcache_update_period,
    )
    n_speakers = diar_model.sortformer_modules.n_spk
    print(f"  n_speakers = {n_speakers}")

    print("Loading DiCoW...")
    dicow, tokenizer, feature_extractor = load_dicow(dicow_model_path, device)

    # ---------------------------------------------------------------
    # Build the online processor
    # ---------------------------------------------------------------
    asr_backend = DiCoWStreamingASR(
        model=dicow,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device,
        language=language,
        target_speaker=TARGET_SPEAKER,
    )

    online = DiCoWOnlineASRProcessor(
        asr=asr_backend,
        n_speakers=n_speakers,
        buffer_trimming=("segment", buffer_trim_sec),
    )

    # ---------------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------------
    print(f"Loading audio: {audio_path}")
    audio_full = librosa.load(audio_path, sr=SAMPLING_RATE, dtype=np.float32)[0]
    duration = len(audio_full) / SAMPLING_RATE
    print(f"  duration = {duration:.2f}s")

    # ---------------------------------------------------------------
    # Sortformer streaming state
    # ---------------------------------------------------------------
    streaming_state = diar_model.sortformer_modules.init_streaming_state(
        batch_size=1, async_streaming=True, device=device
    )
    total_preds = torch.zeros((1, 0, n_speakers), device=device)

    # Chunk size in samples: chunk_len diar-frames × subsampling × feature_stride
    chunk_samples = int(
        chunk_len * SORTFORMER_SUBSAMPLING *
        (SORTFORMER_FEATURE_STRIDE_MS / 1000.0) * SAMPLING_RATE
    )
    print(f"  chunk_samples = {chunk_samples}  ({chunk_samples/SAMPLING_RATE*1000:.0f}ms)")

    # min audio buffer size before DiCoW fires (in samples)
    min_buffer_samples = int(min_chunk_size_s * SAMPLING_RATE)

    # ---------------------------------------------------------------
    # Streaming loop
    # ---------------------------------------------------------------
    pos = 0
    chunk_idx = 0
    prev_mask_frames = 0  # how many mask frames were in total_preds last step

    print("\n--- Streaming started ---")
    while pos < len(audio_full):
        end = min(pos + chunk_samples, len(audio_full))
        audio_chunk = audio_full[pos:end]

        # Pad last chunk to full chunk_samples so Sortformer encoder is happy
        if len(audio_chunk) < chunk_samples:
            audio_chunk = np.pad(audio_chunk, (0, chunk_samples - len(audio_chunk)))

        # ---- 1. Sortformer: one streaming step ----
        audio_tensor  = torch.from_numpy(audio_chunk).unsqueeze(0).to(device)
        chunk_length  = torch.tensor([chunk_samples], dtype=torch.long, device=device)

        with torch.inference_mode():
            processed, proc_len = diar_model.process_signal(
                audio_signal=audio_tensor,
                audio_signal_length=chunk_length,
            )
            streaming_state, total_preds = diar_model.forward_streaming_step(
                processed_signal=processed,
                processed_signal_length=proc_len,
                streaming_state=streaming_state,
                total_preds=total_preds,
                left_offset=0,
                right_offset=0,
            )

        # Extract only the new mask frames produced by this chunk
        cur_mask_frames = total_preds.shape[1]
        new_frames_count = cur_mask_frames - prev_mask_frames
        if new_frames_count > 0:
            new_mask = total_preds[0, prev_mask_frames:, :].T  # [n_spk, new_frames]
            # Binarize at 0.5 threshold
            new_mask_binary = (new_mask > 0.5).float()
            online.insert_mask_chunk(new_mask_binary)
        prev_mask_frames = cur_mask_frames

        # ---- 2. whisper_streaming: feed audio, maybe transcribe ----
        online.insert_audio_chunk(audio_chunk)

        # Only call DiCoW when we have enough audio buffered
        buffer_len = len(online.audio_buffer)
        if buffer_len >= min_buffer_samples:
            output = online.process_iter()
            _print_output(output, chunk_idx)

        pos = end
        chunk_idx += 1

    # ---- Flush remaining hypothesis ----
    print("\n--- Flushing remaining buffer ---")
    output = online.finish()
    _print_output(output, chunk_idx, final=True)

    print("\n--- Streaming complete ---")


def _print_output(output, chunk_idx: int, final: bool = False):
    beg, end, text = output
    if text.strip():
        tag = "FINAL" if final else f"chunk {chunk_idx:04d}"
        beg_s = f"{beg:.2f}" if beg is not None else "?"
        end_s = f"{end:.2f}" if end is not None else "?"
        print(f"[{tag}] {beg_s}s – {end_s}s : {text}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="True streaming Sortformer + DiCoW pipeline")
    parser.add_argument("audio_path",             type=str, help="Path to 16kHz mono WAV file")
    parser.add_argument("--config",               type=str, default="Sortformer/config.ini")
    parser.add_argument("--sortformer_model",     type=str, default=None)
    parser.add_argument("--dicow_model",          type=str, default=None)
    parser.add_argument("--language",             type=str, default="en")
    parser.add_argument("--min_chunk_size",       type=float, default=1.0,
                        help="Min audio buffer (s) before DiCoW fires")
    parser.add_argument("--buffer_trim_sec",      type=float, default=15.0,
                        help="Audio buffer trim threshold (s)")
    parser.add_argument("--target_speaker",       type=int, default=0)
    args = parser.parse_args()

    # Read config.ini for model paths if not provided on CLI
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    sortformer_path = args.sortformer_model or cfg.get("DEFAULT", "SORTFORMER_MODEL_PATH")
    dicow_path      = args.dicow_model      or cfg.get("DEFAULT", "DICOW_MODEL_PATH")

    TARGET_SPEAKER  = args.target_speaker

    run_streaming(
        audio_path           = args.audio_path,
        sortformer_model_path= sortformer_path,
        dicow_model_path     = dicow_path,
        language             = args.language,
        min_chunk_size_s     = args.min_chunk_size,
        buffer_trim_sec      = args.buffer_trim_sec,
        chunk_len            = int(cfg.get("DEFAULT", "CHUNK_SIZE",  fallback="6")),
        fifo_len             = int(cfg.get("DEFAULT", "FIFO_SIZE",   fallback="188")),
        spkcache_len         = int(cfg.get("DEFAULT", "SPEAKER_CACHE_SIZE", fallback="188")),
        spkcache_update_period = int(cfg.get("DEFAULT", "UPDATE_PERIOD", fallback="144")),
    )
