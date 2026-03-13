import os
from typing import Dict, Optional

import torch
import gradio as gr
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from librosa import load as libr_load
from soundfile import write as sf_write


class DiCoW_Pipeline(AutomaticSpeechRecognitionPipeline):

    ############################################
    # STNO MASK
    ############################################
    @staticmethod
    def get_stno_mask(diar_mask, s_index):

        non_target_mask = torch.ones((diar_mask.shape[0],), dtype=torch.bool)
        non_target_mask[s_index] = False

        sil_frames = (1 - diar_mask).prod(axis=0)
        anyone_else = (1 - diar_mask[non_target_mask]).prod(axis=0)

        target_spk = diar_mask[s_index] * anyone_else
        non_target_spk = (1 - diar_mask[s_index]) * (1 - anyone_else)

        overlapping_speech = diar_mask[s_index] - target_spk

        stno_mask = torch.stack(
            [sil_frames, target_spk, non_target_spk, overlapping_speech],
            axis=0
        )

        return stno_mask

    ############################################
    # INIT
    ############################################
    def __init__(self, *args, speaker_embedding_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.speaker_embedding_model = speaker_embedding_model
        self.type = "seq2seq_whisper"

    ############################################
    # PREPROCESS
    ############################################
    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):

        mixed_audio_path = inputs["audio_filepath"]
        # enrollment_audio_path = inputs.get("enrollment_audio_path", None)

        print("+--------------------------------+")
        print("|          Preprocessing         |")
        print("+--------------------------------+")

        ####################################################
        # 1️⃣ Resample audio
        ####################################################
        print("1. Resampling audio to 16kHz...")

        mixed_input_aud, sr = libr_load(mixed_audio_path, sr=16000, mono=True)
        sf_write(mixed_audio_path, mixed_input_aud, sr, format="wav")

        print("input to diarization:", mixed_audio_path)
        print()

        ####################################################
        # 2️⃣ Load diarization mask
        ####################################################
        print("2. Loading diarization mask...")

        # Use in-memory mask if available, otherwise load from disk
        if self.diarization_mask is not None:
            diarization_mask = self.diarization_mask
            print("Using in-memory diarization mask")
        else:
            audio_name = os.path.basename(mixed_audio_path).replace(".wav", "")
            diar_mask_path = os.path.join("diarisation_masks", f"{audio_name}_mask.pt")
            diarization_mask = torch.load(diar_mask_path)
            print(f"Loaded mask from disk: {diar_mask_path}")

        print("diarization_mask shape:", diarization_mask.shape)
        print()

        ####################################################
        # 3️⃣ Run base Whisper preprocessing
        ####################################################
        generator = super().preprocess(
            mixed_audio_path,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s
        )

        samples = next(generator)

        print("samples['input_features'] shape:", samples["input_features"].shape)
        print()

        # Get target speaker ID from inputs or default to 0
        target_speaker_id = inputs.get("target_speaker_id", 0)
        print(f"Target speaker ID: {target_speaker_id}")
        samples["target_speaker_id"] = target_speaker_id

        ####################################################
        # 4️⃣ Create STNO masks
        ####################################################
        print("3. Creating STNO masks...")

        num_speakers = diarization_mask.shape[0]

        stno_masks = []
        for i in range(num_speakers):
            stno_mask = self.get_stno_mask(diarization_mask, i)
            stno_masks.append(stno_mask)
        stno_masks = torch.stack(stno_masks, axis=0)

        print("stno_masks shape:", stno_masks.shape)
        print()

        ####################################################
        # 5️⃣ Expand features for target speaker only
        ####################################################

        samples["stno_mask"] = stno_masks.to(
            samples["input_features"].device,
            dtype=samples["input_features"].dtype
        )

        samples["input_features"] = samples["input_features"].repeat(
            num_speakers,
            1,
            1
        )

        samples["attention_mask"] = torch.ones(
            samples["input_features"].shape[0],
            samples["input_features"].shape[2],
            dtype=torch.bool,
            device=samples["input_features"].device
        )

        ####################################################
        # Optional cleanup
        ####################################################

        if "num_frames" in samples:
            del samples["num_frames"]

        yield samples

    ############################################
    # FORWARD
    ############################################
    def _forward(self, model_inputs, return_timestamps=False, **generate_kwargs):

        print("+--------------------------------+")
        print("|             Forward            |")
        print("+--------------------------------+")

        attention_mask = model_inputs.pop("attention_mask", None)
        stride = model_inputs.pop("stride", None)
        segment_size = model_inputs.pop("segment_size", None)
        is_last = model_inputs.pop("is_last")

        target_speaker_id = model_inputs.pop("target_speaker_id")

        if stride is not None and segment_size is not None:
            raise ValueError("segment_size must be used only when stride is None")

        if "input_features" in model_inputs:
            inputs = model_inputs.pop("input_features")
        elif "input_values" in model_inputs:
            inputs = model_inputs.pop("input_values")
        else:
            raise ValueError(
                "Model requires `input_features` or `input_values`"
            )
        
        # Handle stno_mask if present
        stno_mask = model_inputs.pop("stno_mask", None)

        ############################################
        # SELECT TARGET SPEAKER
        ############################################
        
        # Select the target speaker's features from the batch
        print(f"inputs shape before indexing: {inputs.shape}")
        print(f"target_speaker_id: {target_speaker_id}")
        
        inputs = inputs[target_speaker_id].unsqueeze(0)

        if attention_mask is not None:
            attention_mask = attention_mask[target_speaker_id].unsqueeze(0)
        
        if stno_mask is not None:
            stno_mask = stno_mask[target_speaker_id].unsqueeze(0)
            model_inputs["stno_mask"] = stno_mask

        ############################################
        # WHISPER TIMESTAMP OPTIONS
        ############################################

        if return_timestamps and self.type == "seq2seq_whisper":

            generate_kwargs["return_timestamps"] = return_timestamps

            if return_timestamps == "word":
                generate_kwargs["return_token_timestamps"] = True
                generate_kwargs["return_segments"] = True

            generate_kwargs["input_features"] = inputs

        ############################################
        # GENERATE
        ############################################

        tokens = self.model.generate(
            attention_mask=attention_mask,
            **generate_kwargs,
            **model_inputs,
        )

        return {
            "is_last": is_last,
            "tokens": tokens
        }

    ############################################
    # POSTPROCESS
    ############################################
    def postprocess(
        self,
        model_outputs,
        decoder_kwargs: Optional[Dict] = None,
        return_timestamps=None,
        return_language=None
    ):

        print("+--------------------------------+")
        print("|          Postprocessing        |")
        print("+--------------------------------+")

        per_spk_outputs = self.tokenizer.batch_decode(
            model_outputs[0]['tokens'],
            decode_with_timestamps=True,
            skip_special_tokens=True
        )

        return {
            "text": None,
            "per_spk_outputs": per_spk_outputs
        }