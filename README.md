# Sortformer_Whisper: Speaker Diarization + Transcription Pipeline

End-to-end pipeline for speaker diarization using Sortformer and target-speaker transcription using DiCoW.

## Overview

This pipeline performs 3 steps:
1. **Generate Manifest**: Create dataset manifest from audio file list
2. **Sortformer Diarization**: Run speaker diarization to get speaker masks (in-memory, no temporary files)
3. **DiCoW Transcription**: Transcribe audio for each speaker separately using diarization masks

All speaker transcriptions are output with proper naming: `{audio_name}_speaker{id}.txt`

## Setup

### Environment
```bash
conda create -n socow python=3.11
conda activate socow
pip install -r requirements.txt

pip install 'nemo_toolkit[asr]==2.6.0'
pip install 'transformers==4.42.0' 'tokenizers==0.19.1'
```

### Configuration

Edit `Sortformer/config.ini` to adjust parameters:

```ini
[DEFAULT]
# Streaming parameters for real-time diarization
CHUNK_SIZE=6
RIGHT_CONTEXT=7
FIFO_SIZE=188
UPDATE_PERIOD=144
SPEAKER_CACHE_SIZE=188

# Processing flags
STREAMING=true
BYPASS_POSTPROCESSING=false

# Model paths
SORTFORMER_MODEL_PATH=Sortformer/diar_streaming_sortformer_4spk-v2.1.nemo
DICOW_MODEL_PATH=/path/to/DiCoW/model

# Input/Output paths
MIXED_LIST_PATH=mixed_list.txt
DATASET_MANIFEST=output/dataset_manifest.json
OUTPUT_TRANSCRIPTION_DIR=output
```

## Usage

### Quick Start

```python
from Sortformer_Whisper import Sortformer_Whisper_Pipeline

pipeline = Sortformer_Whisper_Pipeline(config_path="Sortformer/config.ini")
pipeline.run_pipeline()
```

### Step-by-Step

```python
pipeline = Sortformer_Whisper_Pipeline(config_path="Sortformer/config.ini")

# Step 1: Generate manifest from mixed audio list
pipeline.step1_generate_manifest("mixed_list.txt", "output/dataset_manifest.json")

# Step 2: Get diarization masks (in memory)
masks = pipeline.step2_run_diarization("output/dataset_manifest.json")

# Step 3: Transcribe each speaker
pipeline.step3_run_transcription("output/dataset_manifest.json", masks, "output")
```

### Command Line

```bash
python run.py
```

## Input Format

### mixed_list.txt
Simple text file with one audio file path per line:
```
/path/to/audio1.wav
/path/to/audio2.wav
```

## Output Format

For each audio file, separate transcription files are created for each speaker:
```
output/
├── dataset_manifest.json
├── audio1_speaker0.txt
├── audio1_speaker1.txt
├── audio2_speaker0.txt
└── audio2_speaker1.txt
```

## Architecture

### Classes

- **ManifestGenerator**: Converts audio file list to manifest format
- **SortformerDiarizer**: Runs Sortformer diarization, returns in-memory masks
- **DiCoWTranscriber**: Per-speaker transcription using DiCoW model
- **Sortformer_Whisper_Pipeline**: Main orchestrator handling the 3-step pipeline

### Data Flow

```
mixed_list.txt
    ↓
ManifestGenerator (step 1)
    ↓
dataset_manifest.json
    ↓ + audio files
SortformerDiarizer (step 2)
    ↓
Dict[str, torch.Tensor] masks (in memory)
    ↓ + audio files
DiCoWTranscriber (step 3)
    ↓
speaker0.txt, speaker1.txt, ...
```

## File Structure

```
├── Sortformer/
│   ├── config.ini                       ← All configuration parameters
│   ├── diar_streaming_sortformer_4spk-v2.1.nemo  ← Diarization model
│   ├── generate_manifest.py             ← Step 1: Generate manifest
│   ├── get_diarisation_mask.py          ← Step 2: Run diarization
│   └── dicow_inference.py               ← Step 3: Transcription
│
├── model/
│   └── DiCoW/                           ← DiCoW transcription model
│
├── Sortformer_Whisper.py                ← Main orchestrator
├── run.py                               ← CLI entry point
├── mixed_list.txt                       ← Input: audio file paths
└── output/                              ← Generated outputs
    ├── dataset_manifest.json
    └── *_speaker*.txt
```

## Key Features

- ✅ In-memory diarization masks (no intermediate `.pt` files)
- ✅ Per-speaker transcription with speaker ID in filename
- ✅ Streaming diarization support for real-time processing
- ✅ Configurable parameters via `config.ini`
- ✅ Class-based architecture for easy integration

## Requirements

- PyTorch
- NeMo (for Sortformer diarization)
- Transformers (for DiCoW transcription)
- librosa (audio processing)
- omegaconf (configuration management)
