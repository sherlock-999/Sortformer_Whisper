"""
StreamingSortformer: chunk-by-chunk Sortformer diarization.

Exposes a simple two-method API:
    sorformer = StreamingSortformer(cfg)
    sortformer.reset()                     # call once per audio file
    mask_chunk = sortformer.process_chunk(audio_chunk)  # call per 480ms chunk

Speaker cache, FIFO queue, and all streaming state are maintained internally.
The caller only ever sees the binary diarization mask for the latest chunk.
"""

import numpy as np
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models import SortformerEncLabelModel


class StreamingSortformer:
    """
    Wraps SortformerEncLabelModel with a stateful chunk-by-chunk inference API.

    All streaming internals (speaker cache, FIFO queue, permutation tracking)
    are hidden inside self.streaming_state and updated automatically on each
    call to process_chunk().  Speaker identity is therefore consistent across
    all chunks of the same audio file without any external bookkeeping.

    Args:
        cfg: DictConfig with the following keys (see config.yaml [sortformer]):
            model_path             – path to .nemo streaming checkpoint
            chunk_len              – chunk length in diarization frames (default 6)
            fifo_len               – FIFO queue length in frames (default 188)
            spkcache_len           – speaker cache length in frames (default 188)
            spkcache_update_period – how often cache is updated (default 144)
            pred_threshold         – sigmoid threshold for binarising mask (default 0.5)
    """

    # Audio constants (must match Sortformer model training config)
    SAMPLING_RATE      = 16000
    SUBSAMPLING_FACTOR = 8      # Fast-Conformer encoder subsampling
    FEATURE_STRIDE_S   = 0.01   # 10ms mel frame stride

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pred_threshold = float(cfg.get("pred_threshold", 0.5))

        print(f"[StreamingSortformer] Loading model from {cfg.model_path} ...")
        self._model = SortformerEncLabelModel.restore_from(
            restore_path=cfg.model_path,
            map_location=self.device,
        ).eval().to(self.device)

        if not self._model.streaming_mode:
            raise RuntimeError(
                "Sortformer model must have streaming_mode=True. "
                "Use the streaming .nemo checkpoint."
            )

        # Apply config to model streaming parameters
        self._model.async_streaming = True
        self._model.sortformer_modules.chunk_len               = int(cfg.chunk_len)
        self._model.sortformer_modules.fifo_len                = int(cfg.fifo_len)
        self._model.sortformer_modules.spkcache_len            = int(cfg.spkcache_len)
        self._model.sortformer_modules.spkcache_update_period  = int(cfg.spkcache_update_period)
        self._model.sortformer_modules.chunk_left_context      = 0   # no look-back: true online
        self._model.sortformer_modules.chunk_right_context     = 0   # no look-ahead: true online

        # Derived constants
        self.n_speakers    = self._model.sortformer_modules.n_spk
        self.chunk_samples = int(
            cfg.chunk_len * self.SUBSAMPLING_FACTOR * self.FEATURE_STRIDE_S * self.SAMPLING_RATE
        )

        print(f"[StreamingSortformer] Ready. "
              f"n_speakers={self.n_speakers}, "
              f"chunk_samples={self.chunk_samples} "
              f"({self.chunk_samples / self.SAMPLING_RATE * 1000:.0f} ms per chunk)")

        # Internal streaming state — initialised by reset()
        self._streaming_state = None
        self._total_preds     = None

    # ------------------------------------------------------------------
    def reset(self):
        """
        Initialise (or re-initialise) streaming state for a new audio file.
        Must be called before the first process_chunk() of every new file.
        """
        self._streaming_state = self._model.sortformer_modules.init_streaming_state(
            batch_size=1, async_streaming=True, device=self.device
        )
        self._total_preds = torch.zeros((1, 0, self.n_speakers), device=self.device)

    # ------------------------------------------------------------------
    def process_chunk(self, audio_chunk: np.ndarray, true_length: int = None) -> torch.Tensor:
        """
        Feed one fixed-length audio chunk and return its diarization mask.

        Args:
            audio_chunk:  float32 numpy array of exactly self.chunk_samples samples.
                          The caller is responsible for padding the final chunk if needed.
            true_length:  number of real (un-padded) samples in audio_chunk.
                          Must be provided when the last chunk is shorter than chunk_samples
                          so that the model's length tracking matches streaming_feat_loader
                          behaviour (which passes actual mel-frame counts, not padded counts).
                          Defaults to len(audio_chunk) when not provided.

        Returns:
            preds: float32 tensor of shape [n_speakers, new_frames] on CPU.
                   Raw sigmoid probabilities in [0, 1].
                   new_frames is typically cfg.chunk_len but may vary near file end.
        """
        if self._streaming_state is None:
            raise RuntimeError("Call reset() before process_chunk().")

        prev_n_frames = self._total_preds.shape[1]

        if true_length is None:
            true_length = len(audio_chunk)

        audio_t  = torch.from_numpy(audio_chunk).unsqueeze(0).to(self.device)  # [1, samples]
        length_t = torch.tensor([true_length], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            processed, proc_len = self._model.process_signal(
                audio_signal=audio_t,
                audio_signal_length=length_t,
            )
            # process_signal returns (1, feat_dim, time); forward_streaming_step / pre_encode
            # expects (1, time, feat_dim) — same transpose that streaming_feat_loader applies.
            processed = processed.transpose(1, 2)
            self._streaming_state, self._total_preds = self._model.forward_streaming_step(
                processed_signal=processed,
                processed_signal_length=proc_len,
                streaming_state=self._streaming_state,
                total_preds=self._total_preds,
                left_offset=0,
                right_offset=0,
            )

        # Extract only the new frames: [n_spk, new_frames], raw sigmoid values.
        # Caller is responsible for post-processing (thresholding / onset-offset hysteresis).
        new_frames = self._total_preds[0, prev_n_frames:, :].T.cpu()  # [n_spk, new_frames]
        return new_frames
