from Sortformer_Whisper import Sortformer_Whisper_Pipeline

pipeline = Sortformer_Whisper_Pipeline(config_path="Sortformer/config.ini")
pipeline.run_pipeline(
    mixed_list_path="mixed_list.txt",
    output_dir="output_transcriptions/"
)