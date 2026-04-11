#!/usr/bin/env python3
"""
Gradio streaming demo for Sortformer_Whisper_Streaming.

Records from the microphone in real time, feeds 480ms chunks to Sortformer for
diarization, and calls DiCoW every `transcription_interval_s` seconds to produce
a live diarised transcript.
"""

import re
import numpy as np
import torch
import gradio as gr
from math import gcd

from Sortformer_Whisper_Streaming import Sortformer_Whisper_Streaming_Pipeline
from dicow_inference import DiCoWTranscriber

# ---------------------------------------------------------------------------
# Constants / globals
# ---------------------------------------------------------------------------

SAMPLING_RATE = 16000

COLORS       = ["#4A90D9", "#E07B39", "#4CAF50", "#9C27B0"]
LIGHT_COLORS = ["#D6E8F7", "#FAE3D0", "#D4EDDA", "#E8D5F5"]

# Load pipeline once — models stay in memory for the lifetime of the server.
print("Loading Sortformer_Whisper_Streaming pipeline …")
pipeline = Sortformer_Whisper_Streaming_Pipeline(config_path="config.yaml")
INTERVAL_S = pipeline.cfg.pipeline.transcription_interval_s
print(f"Pipeline ready. Transcription interval: {INTERVAL_S}s\n")


# ---------------------------------------------------------------------------
# Stream state helpers
# ---------------------------------------------------------------------------

def make_fresh_state() -> dict:
    """Return a blank per-session state dict."""
    return {
        # True once sortformer.reset() has been called for this recording session
        "initialized":    False,
        # Raw audio arriving from the browser, not yet sliced into Sortformer chunks
        "raw_buffer":     np.zeros(0, dtype=np.float32),
        # Audio that has been through Sortformer but not yet sent to DiCoW
        "accum_audio":    np.zeros(0, dtype=np.float32),
        # Mask co-buffer aligned with accum_audio
        "accum_mask":     None,
        # Completed transcript segments (list of dicts)
        "all_segments":   [],
        # Absolute time offset for the current transcription window
        "window_start_s": 0.0,
    }


def resample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample `audio` from `orig_sr` to SAMPLING_RATE using integer ratio."""
    if orig_sr == SAMPLING_RATE:
        return audio
    try:
        from scipy.signal import resample_poly
        g    = gcd(orig_sr, SAMPLING_RATE)
        up   = SAMPLING_RATE // g
        down = orig_sr // g
        return resample_poly(audio, up, down).astype(np.float32)
    except ImportError:
        # Fallback: linear interpolation (lower quality)
        n_out = int(len(audio) * SAMPLING_RATE / orig_sr)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# HTML rendering (reused from gradio_demo.py style)
# ---------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = seconds % 60
    return f"{m}:{s:05.2f}"


def _speaker_idx(speaker: str, order: list) -> int:
    if speaker not in order:
        order.append(speaker)
    return order.index(speaker)


def build_timeline_html(segments: list, total_duration: float) -> str:
    if not segments or total_duration <= 0:
        return '<div style="color:#888;font-family:sans-serif;padding:8px;">Recording… no segments yet.</div>'

    speaker_order: list = []
    for seg in segments:
        _speaker_idx(seg["speaker"], speaker_order)

    rows = ""
    for i, spk in enumerate(speaker_order):
        color = COLORS[i % len(COLORS)]
        label = spk.replace("_", " ").title()
        bars  = ""
        for seg in segments:
            if seg["speaker"] != spk:
                continue
            left  = seg["start_time"] / total_duration * 100
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


def build_chat_html(segments: list) -> str:
    if not segments:
        return '<div style="color:#888;font-family:sans-serif;padding:8px;">Waiting for first transcript…</div>'

    speaker_order: list = []
    for seg in segments:
        _speaker_idx(seg["speaker"], speaker_order)

    bubbles = ""
    for seg in segments:
        words = seg.get("words", "").strip()
        if not words:
            continue
        idx   = speaker_order.index(seg["speaker"])
        color = COLORS[idx % len(COLORS)]
        bg    = LIGHT_COLORS[idx % len(LIGHT_COLORS)]
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
        f'  <div style="font-weight:700;margin-bottom:10px;font-size:14px;">Live Transcript</div>'
        f'  {bubbles}'
        f'</div>'
    )


def build_status_html(state: dict) -> str:
    n_seg       = len(state["all_segments"])
    elapsed     = state["window_start_s"] + len(state["accum_audio"]) / SAMPLING_RATE
    buf_pending = len(state["raw_buffer"]) / SAMPLING_RATE
    color       = "#4CAF50" if state["initialized"] else "#aaa"
    dot         = "🟢" if state["initialized"] else "⚪"
    return (
        f'<div style="font-family:monospace;font-size:12px;color:#555;padding:6px 0;">'
        f'{dot} &nbsp; elapsed: <b>{elapsed:.1f}s</b> &nbsp;|&nbsp; '
        f'segments: <b>{n_seg}</b> &nbsp;|&nbsp; '
        f'buffer pending: <b>{buf_pending*1000:.0f}ms</b>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Core streaming callback
# ---------------------------------------------------------------------------

def _run_transcription(state: dict) -> None:
    """Call DiCoW on the current accumulation window and append results to state."""
    window_end_s = state["window_start_s"] + len(state["accum_audio"]) / SAMPLING_RATE

    print(f"  → Transcribing {state['window_start_s']:.1f}s – {window_end_s:.1f}s  "
          f"(mask {state['accum_mask'].shape})")

    pipeline.transcriber.pipeline.diarization_mask = state["accum_mask"]
    result = pipeline.transcriber.pipeline(
        {"audio": state["accum_audio"]},
        return_timestamps=True,
    )
    pipeline.transcriber.pipeline.diarization_mask = None

    for spk_idx, spk_text in enumerate(result.get("per_spk_outputs", [])):
        if not spk_text or not spk_text.strip():
            continue
        segments = DiCoWTranscriber._extract_segments_with_timing(spk_text)
        for (seg_start, seg_end, text) in segments:
            if not text.strip():
                continue
            state["all_segments"].append({
                "speaker":    f"speaker_{spk_idx}",
                "start_time": round(state["window_start_s"] + seg_start, 3),
                "end_time":   round(state["window_start_s"] + seg_end,   3),
                "words":      text,
            })

    state["window_start_s"] = window_end_s
    state["accum_audio"]    = np.zeros(0, dtype=np.float32)
    state["accum_mask"]     = None


def process_stream_chunk(chunk, state: dict):
    """
    Gradio .stream() callback.

    Receives one microphone chunk as (sample_rate, np.ndarray), feeds it through
    Sortformer in 480ms steps, and transcribes with DiCoW every INTERVAL_S seconds.
    Returns updated (timeline_html, chat_html, status_html, state).
    """
    if chunk is None:
        return (
            build_timeline_html(state["all_segments"],
                                state["window_start_s"] + len(state["accum_audio"]) / SAMPLING_RATE),
            build_chat_html(state["all_segments"]),
            build_status_html(state),
            state,
        )

    sr, audio_np = chunk

    # ---- Initialise Sortformer for this recording session ----
    if not state["initialized"]:
        pipeline.sortformer.reset()
        state["initialized"] = True

    # ---- Convert to float32 mono ----
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype(np.float32)
    if np.abs(audio_np).max() > 1.5:        # likely int16 from browser
        audio_np /= 32768.0

    # ---- Resample to 16 kHz ----
    audio_16k = resample(audio_np, sr)

    # ---- Append to raw buffer ----
    state["raw_buffer"] = np.concatenate([state["raw_buffer"], audio_16k])

    chunk_samples    = pipeline.sortformer.chunk_samples
    interval_samples = int(INTERVAL_S * SAMPLING_RATE)

    # ---- Drain raw buffer in exact Sortformer chunks ----
    while len(state["raw_buffer"]) >= chunk_samples:
        true_audio           = state["raw_buffer"][:chunk_samples]
        state["raw_buffer"]  = state["raw_buffer"][chunk_samples:]

        mask_chunk = pipeline.sortformer.process_chunk(true_audio)

        state["accum_audio"] = np.concatenate([state["accum_audio"], true_audio])
        state["accum_mask"]  = (
            mask_chunk
            if state["accum_mask"] is None
            else torch.cat([state["accum_mask"], mask_chunk], dim=1)
        )

    # ---- Transcribe when interval is full ----
    if len(state["accum_audio"]) >= interval_samples:
        _run_transcription(state)

    total_duration = state["window_start_s"] + len(state["accum_audio"]) / SAMPLING_RATE
    return (
        build_timeline_html(state["all_segments"], total_duration),
        build_chat_html(state["all_segments"]),
        build_status_html(state),
        state,
    )


def flush_and_transcribe(state: dict):
    """
    Force-transcribe whatever audio remains in the accumulation buffer.
    Called when the user clicks "Finalize" after stopping the microphone.
    """
    # Pad any remaining raw_buffer fragment to a full Sortformer chunk
    if len(state["raw_buffer"]) > 0 and state["initialized"]:
        chunk_samples = pipeline.sortformer.chunk_samples
        true_audio    = state["raw_buffer"]
        padded        = np.pad(true_audio, (0, chunk_samples - len(true_audio)))
        mask_chunk    = pipeline.sortformer.process_chunk(padded)

        state["accum_audio"] = np.concatenate([state["accum_audio"], true_audio])
        state["accum_mask"]  = (
            mask_chunk
            if state["accum_mask"] is None
            else torch.cat([state["accum_mask"], mask_chunk], dim=1)
        )
        state["raw_buffer"] = np.zeros(0, dtype=np.float32)

    # Transcribe remaining accumulation (even if shorter than INTERVAL_S)
    if len(state["accum_audio"]) > 0 and state["accum_mask"] is not None:
        _run_transcription(state)

    total_duration = state["window_start_s"]
    return (
        build_timeline_html(state["all_segments"], total_duration),
        build_chat_html(state["all_segments"]),
        build_status_html(state),
        state,
    )


def clear_all(_state):
    """Reset everything — called by the Clear button."""
    fresh = make_fresh_state()
    empty = '<div style="color:#888;font-family:sans-serif;padding:8px;">Cleared.</div>'
    return fresh, empty, empty, empty


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Sortformer Whisper — Live Streaming") as demo:

    gr.Markdown("## Sortformer + DiCoW — Real-Time Streaming Demo")
    gr.Markdown(
        "Click **Record** to start the microphone. "
        f"DiCoW transcribes every **{INTERVAL_S}s** of accumulated audio. "
        "Click **Finalize** when done to flush any remaining audio."
    )

    state = gr.State(make_fresh_state())

    with gr.Row():
        audio_in = gr.Audio(
            sources=["microphone"],
            streaming=True,
            label="Microphone Input",
        )

    status_out   = gr.HTML(label="Status")
    timeline_out = gr.HTML(label="Speaker Timeline")
    chat_out     = gr.HTML(label="Live Transcript")

    with gr.Row():
        finalize_btn = gr.Button("Finalize (flush remaining audio)", variant="primary")
        clear_btn    = gr.Button("Clear", variant="secondary")

    # Stream every incoming chunk to process_stream_chunk
    audio_in.stream(
        fn=process_stream_chunk,
        inputs=[audio_in, state],
        outputs=[timeline_out, chat_out, status_out, state],
        stream_every=0.5,       # fire callback every ~0.5s
        time_limit=None,        # no hard cap on recording length
    )

    finalize_btn.click(
        fn=flush_and_transcribe,
        inputs=[state],
        outputs=[timeline_out, chat_out, status_out, state],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[state],
        outputs=[state, timeline_out, chat_out, status_out],
    )


if __name__ == "__main__":
    demo.launch(share=True)
