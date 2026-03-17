from Sortformer_Whisper import Sortformer_Whisper_Pipeline

pipeline = Sortformer_Whisper_Pipeline(config_path="config.ini")

pipeline.run_pipeline( 
    mixed_list_path="/home/hesiyuan/Desktop/sortformer_test/testset/audio_list/l2m_audio_list.txt",
    output_dir="output_transcriptions/",
    output_filename = "l2m_hypothesis_multi.jsonl"
)
'''
pipeline.run_pipeline( 
    mixed_list_path="/home/hesiyuan/Desktop/sortformer_test/Sortformer_Whisper/mixed_list.txt",
    output_dir="output_transcriptions/",
    output_filename = "mixed_hypothesis_multi.jsonl"
)
'''