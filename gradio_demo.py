import gradio as gr
import os
from Sortformer_Whisper import Sortformer_Whisper_Pipeline
import tempfile

# Initialize pipeline (loads config and models)
pipeline = Sortformer_Whisper_Pipeline(config_path="config.ini")

def transcribe_audio(audio_file_path):
    # audio_file_path is a string path to the uploaded file
    tmp_path = audio_file_path
    # Prepare a single-item mixed_list.txt for pipeline
    mixed_list_path = tmp_path + "_list.txt"
    with open(mixed_list_path, "w") as f:
        f.write(tmp_path + "\n")
    # Output directory for results
    output_dir = tempfile.mkdtemp()
    output_filename = "gradio_hypothesis_multi.jsonl"
    # Run pipeline (steps: manifest, diarization, transcription)
    pipeline.run_pipeline(mixed_list_path=mixed_list_path, output_dir=output_dir, output_filename=output_filename)
    # Read result
    result_path = os.path.join(output_dir, output_filename)
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            result = f.read()
    else:
        result = "No output generated."
    return result

demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio (.wav)"),
    outputs=gr.Textbox(label="Transcription Output (JSONL)"),
    title="Sortformer Whisper DiCoW Pipeline",
    description="Upload a .wav file to run diarization and transcription using the Sortformer_Whisper pipeline."
)

if __name__ == "__main__":
    demo.launch()
