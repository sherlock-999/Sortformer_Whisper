import gradio as gr
import os
import json
import wave
import tempfile
from pathlib import Path
from Sortformer_Whisper import Sortformer_Whisper_Pipeline

# Initialize pipeline (loads config and models)
pipeline = Sortformer_Whisper_Pipeline(config_path="config.ini")

COLORS = ["#4A90D9", "#E07B39", "#4CAF50", "#9C27B0"]
LIGHT_COLORS = ["#D6E8F7", "#FAE3D0", "#D4EDDA", "#E8D5F5"]


def fmt_time(seconds):
    m = int(seconds) // 60
    s = seconds % 60
    return f"{m}:{s:05.2f}"


def parse_segments(jsonl_text):
    segments = []
    for line in jsonl_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            segments.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    segments.sort(key=lambda x: x.get("start_time", 0))
    return segments


def speaker_index(speaker_id, speaker_order):
    if speaker_id not in speaker_order:
        speaker_order.append(speaker_id)
    return speaker_order.index(speaker_id)


def build_timeline_html(segments, total_duration):
    if total_duration <= 0:
        return ""

    speaker_order = []
    for seg in segments:
        speaker_index(seg["speaker"], speaker_order)

    rows = ""
    for i, spk in enumerate(speaker_order):
        color = COLORS[i % len(COLORS)]
        label = spk.replace("_", " ").title()
        bars = ""
        for seg in segments:
            if seg["speaker"] != spk:
                continue
            left = seg["start_time"] / total_duration * 100
            width = max((seg["end_time"] - seg["start_time"]) / total_duration * 100, 0.3)
            bars += (
                f'<div title="{fmt_time(seg["start_time"])}–{fmt_time(seg["end_time"])}: {seg["words"]}" '
                f'style="position:absolute;left:{left:.2f}%;width:{width:.2f}%;height:100%;'
                f'background:{color};border-radius:3px;cursor:pointer;"></div>'
            )
        rows += (
            f'<div style="display:flex;align-items:center;margin-bottom:6px;">'
            f'  <div style="width:90px;font-size:12px;font-weight:600;color:{color};flex-shrink:0;">{label}</div>'
            f'  <div style="position:relative;flex:1;height:18px;background:#e8e8e8;border-radius:3px;">{bars}</div>'
            f'</div>'
        )

    # Time axis
    ticks = ""
    for pct in range(0, 101, 10):
        t = total_duration * pct / 100
        ticks += (
            f'<div style="position:absolute;left:{pct}%;transform:translateX(-50%);'
            f'font-size:10px;color:#888;">{fmt_time(t)}</div>'
        )

    return (
        f'<div style="font-family:sans-serif;padding:12px 0;">'
        f'  <div style="font-weight:700;margin-bottom:10px;font-size:14px;">Speaker Timeline</div>'
        f'  {rows}'
        f'  <div style="position:relative;height:16px;margin-left:90px;margin-top:4px;">{ticks}</div>'
        f'</div>'
    )


def build_chat_html(segments):
    speaker_order = []
    for seg in segments:
        speaker_index(seg["speaker"], speaker_order)

    bubbles = ""
    for seg in segments:
        words = seg.get("words", "").strip()
        if not words:
            continue
        idx = speaker_order.index(seg["speaker"])
        color = COLORS[idx % len(COLORS)]
        bg = LIGHT_COLORS[idx % len(LIGHT_COLORS)]
        label = seg["speaker"].replace("_", " ").title()
        align = "flex-end" if idx % 2 == 1 else "flex-start"
        time_str = f"{fmt_time(seg['start_time'])} – {fmt_time(seg['end_time'])}"
        bubbles += (
            f'<div style="display:flex;flex-direction:column;align-items:{align};margin-bottom:12px;">'
            f'  <div style="font-size:11px;color:{color};font-weight:600;margin-bottom:3px;">{label}</div>'
            f'  <div style="max-width:70%;background:{bg};border:1px solid {color};border-radius:12px;'
            f'       padding:8px 14px;font-size:14px;line-height:1.5;">{words}</div>'
            f'  <div style="font-size:10px;color:#aaa;margin-top:3px;">{time_str}</div>'
            f'</div>'
        )

    return (
        f'<div style="font-family:sans-serif;padding:12px 0;">'
        f'  <div style="font-weight:700;margin-bottom:10px;font-size:14px;">Transcript</div>'
        f'  {bubbles}'
        f'</div>'
    )


def mask_to_segments(mask_1d, fps=50):
    """Convert a binary 1D mask tensor to a list of (start_sec, end_sec) segments."""
    segments = []
    in_seg = False
    start = 0
    values = mask_1d.tolist()
    for i, v in enumerate(values):
        if v > 0.5 and not in_seg:
            start = i / fps
            in_seg = True
        elif v <= 0.5 and in_seg:
            segments.append((start, i / fps))
            in_seg = False
    if in_seg:
        segments.append((start, len(values) / fps))
    return segments


def build_diarization_html(masks, audio_duration):
    """Render a Sortformer diarization timeline from raw mask tensors."""
    if not masks or audio_duration is None or audio_duration <= 0:
        return ""

    # masks is Dict[audio_name -> Tensor[num_speakers, num_frames]]
    # For the Gradio demo there is always exactly one audio file
    mask = next(iter(masks.values()))  # shape [num_speakers, num_frames]
    num_speakers = mask.shape[0]

    rows = ""
    for i in range(num_speakers):
        color = COLORS[i % len(COLORS)]
        label = f"Speaker {i}"
        segments = mask_to_segments(mask[i], fps=50)
        bars = ""
        for seg_start, seg_end in segments:
            left = seg_start / audio_duration * 100
            width = max((seg_end - seg_start) / audio_duration * 100, 0.3)
            bars += (
                f'<div title="{fmt_time(seg_start)} – {fmt_time(seg_end)}" '
                f'style="position:absolute;left:{left:.2f}%;width:{width:.2f}%;height:100%;'
                f'background:{color};border-radius:3px;cursor:pointer;"></div>'
            )
        rows += (
            f'<div style="display:flex;align-items:center;margin-bottom:6px;">'
            f'  <div style="width:90px;font-size:12px;font-weight:600;color:{color};flex-shrink:0;">{label}</div>'
            f'  <div style="position:relative;flex:1;height:18px;background:#e8e8e8;border-radius:3px;">{bars}</div>'
            f'</div>'
        )

    ticks = ""
    for pct in range(0, 101, 10):
        t = audio_duration * pct / 100
        ticks += (
            f'<div style="position:absolute;left:{pct}%;transform:translateX(-50%);'
            f'font-size:10px;color:#888;">{fmt_time(t)}</div>'
        )

    return (
        f'<div style="font-family:sans-serif;padding:12px 0;">'
        f'  <div style="font-weight:700;margin-bottom:10px;font-size:14px;">Sortformer Diarization</div>'
        f'  {rows}'
        f'  <div style="position:relative;height:16px;margin-left:90px;margin-top:4px;">{ticks}</div>'
        f'</div>'
    )


def get_audio_duration(audio_path):
    try:
        with wave.open(audio_path, "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def format_output(jsonl_text, audio_duration=None):
    segments = parse_segments(jsonl_text)
    if not segments:
        empty = '<div style="color:#888;font-family:sans-serif;">No output produced.</div>'
        return empty, empty
    # Use real audio duration so Whisper's over-extended timestamps don't inflate the timeline
    total_duration = audio_duration or max(s.get("end_time", 0) for s in segments)
    # Clamp segment end times to actual audio duration so bars don't overflow
    if audio_duration:
        for seg in segments:
            seg["end_time"] = min(seg["end_time"], audio_duration)
            seg["start_time"] = min(seg["start_time"], audio_duration)
    return build_timeline_html(segments, total_duration), build_chat_html(segments)


def transcribe_audio(audio_file_path):
    if audio_file_path is None:
        empty = '<div style="color:#888;font-family:sans-serif;">Please upload an audio file.</div>'
        return empty, empty, empty

    output_dir     = tempfile.mkdtemp()
    output_filename = "gradio_hypothesis_multi.jsonl"

    # Write a temporary audio list file for the pipeline
    mixed_list_path = os.path.join(output_dir, "audio_list.txt")
    with open(mixed_list_path, "w") as f:
        f.write(audio_file_path + "\n")

    masks = pipeline.run_pipeline(
        mixed_list_path=mixed_list_path,
        output_dir=output_dir,
        output_filename=output_filename,
    )

    audio_duration = get_audio_duration(audio_file_path)
    diar_html      = build_diarization_html(masks, audio_duration)

    result_path = Path(output_dir) / output_filename
    if result_path.exists():
        raw = result_path.read_text(encoding="utf-8")
        timeline_html, chat_html = format_output(raw, audio_duration)
        return diar_html, timeline_html, chat_html

    empty = '<div style="color:#888;font-family:sans-serif;">No output generated.</div>'
    return diar_html, empty, empty


with gr.Blocks(title="Sortformer Whisper DiCoW") as demo:
    gr.Markdown("## Sortformer Whisper DiCoW Pipeline")
    gr.Markdown("Upload a `.wav` file to run speaker diarization and transcription.")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio (.wav)")

    run_btn = gr.Button("Transcribe", variant="primary")

    diar_out = gr.HTML(label="Diarization")
    timeline_out = gr.HTML(label="Speaker Timeline")
    chat_out = gr.HTML(label="Transcript")

    run_btn.click(fn=transcribe_audio, inputs=audio_input, outputs=[diar_out, timeline_out, chat_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
