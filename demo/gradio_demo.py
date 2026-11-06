# coding=utf-8
# Copyright 2026 The RAON Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradio demo for RAON model loaded locally via RaonPipeline.

Usage:
    python demo/gradio_demo.py --model-path /path/to/raon-model
    python demo/gradio_demo.py --model-path KRAFTON/Raon-Speech-9B --hf-token <TOKEN>
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import soundfile as sf
from huggingface_hub import login
from transformers import AutoConfig, AutoModel

try:
    from raon import RaonPipeline
    from raon.models.raon import RaonConfig, RaonModel
except ImportError:
    # raon package not installed — load the model code (not the weights) from HuggingFace Hub
    import importlib

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    _hub = importlib.import_module(
        get_class_from_dynamic_module("modeling_raon.RaonModel", "KRAFTON/Raon-Speech-9B").__module__
    )
    RaonPipeline = _hub.RaonPipeline
    RaonConfig = _hub.RaonConfig
    RaonModel = _hub.RaonModel
    del _hub

# ---------------------------------------------------------------------------
# Example data helpers
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "speechllm" / "eval"
_REPO_ROOT = Path(__file__).parent.parent
_HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")


def _infer_example_language(*texts: str) -> str:
    return "Korean" if any(_HANGUL_RE.search(text) for text in texts) else "English"


def _resolve_data_path(path: str) -> str | None:
    if not path:
        return None
    resolved = (_REPO_ROOT / path).resolve()
    if resolved.exists():
        return str(resolved)
    return None


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_examples(data_dir: Path = _DEFAULT_DATA_DIR) -> list[list]:
    """Build Gradio example rows from eval sample data.

    Each row is ``[task, text, audio, ref_audio, ref_text]``.
    """
    if not data_dir.exists():
        return []

    examples: list[list] = []
    for sample in _load_jsonl_records(data_dir / "stt.jsonl"):
        audio_path = _resolve_data_path(sample.get("audios", [None])[0])
        if audio_path is None:
            continue
        transcript = str(sample.get("conversations", [{}])[-1].get("value", ""))
        examples.append(["STT", "", audio_path, None, ""])

    seen_tts_languages: set[str] = set()
    for sample in _load_jsonl_records(data_dir / "tts.jsonl"):
        conversations = sample.get("conversations", [])
        if not conversations:
            continue
        text = str(conversations[0].get("value", ""))
        language = _infer_example_language(text)
        if language in seen_tts_languages:
            continue
        speaker_refs = sample.get("speaker_ref_audios") or sample.get("audios") or []
        ref_audio_path = _resolve_data_path(str(speaker_refs[0])) if speaker_refs else None
        if ref_audio_path is None:
            continue
        ref_text = (
            "앨리스의 역사 지식에 의하면 그 생쥐가 프랑스 생쥐라는 것보다 이 상황을 설명하는 더 확실한 해명은 없었거든요."
            if language == "Korean"
            else "Villefort rose, half ashamed of being surprised in such a paroxysm of grief."
        )
        examples.append(["TTS", text, None, ref_audio_path, ref_text])
        seen_tts_languages.add(language)

    for sample in _load_jsonl_records(data_dir / "textqa.jsonl"):
        conversations = sample.get("conversations", [])
        audio_path = _resolve_data_path(sample.get("audios", [None])[0])
        if len(conversations) < 2 or audio_path is None:
            continue
        question = str(conversations[-2].get("value", ""))
        examples.append(["TextQA", question, audio_path, None, ""])

    for sample in _load_jsonl_records(data_dir / "speech-chat.jsonl"):
        audio_path = _resolve_data_path(sample.get("audios", [None])[0])
        if audio_path is None:
            continue
        answer = str(sample.get("conversations", [{}])[-1].get("value", ""))
        examples.append(["SpeechChat", "", audio_path, None, ""])

    return examples


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


_temp_audio_files: list[str] = []


def audio_to_tempfile(audio_data: tuple[int, np.ndarray]) -> str:
    """Write Gradio audio to a temporary WAV file, return path."""
    sample_rate, audio_np = audio_data
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]
    # Clean up previous temp files to prevent disk accumulation.
    for old_path in _temp_audio_files:
        try:
            Path(old_path).unlink(missing_ok=True)
        except OSError:
            pass
    _temp_audio_files.clear()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_np, sample_rate)
    _temp_audio_files.append(tmp.name)
    return tmp.name


def tensor_to_gradio_audio(audio_tensor: Any, sr: int) -> tuple[int, np.ndarray]:
    """Convert a torch tensor or numpy array to Gradio audio format."""
    if hasattr(audio_tensor, "detach"):
        audio_np = audio_tensor.detach().cpu().float().numpy()
    elif hasattr(audio_tensor, "numpy"):
        audio_np = audio_tensor.numpy()
    else:
        audio_np = np.asarray(audio_tensor, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np[0]
    return sr, audio_np.astype(np.float32)


# ---------------------------------------------------------------------------
# Inference dispatcher
# ---------------------------------------------------------------------------


def run_inference(
    pipe: RaonPipeline,
    task: str,
    text: str,
    audio: tuple[int, np.ndarray] | None,
    ref_audio: tuple[int, np.ndarray] | None,
    continuation: bool = False,
    ref_text: str = "",
) -> tuple[str, tuple[int, np.ndarray] | None]:
    """Route to the appropriate pipeline method based on task type."""
    try:
        if task == "STT":
            if audio is None:
                return "STT requires audio input.", None
            path = audio_to_tempfile(audio)
            transcript = pipe.stt(path)
            return transcript, None

        elif task == "TTS":
            if not text.strip():
                return "TTS requires text input.", None
            speaker_path = audio_to_tempfile(ref_audio) if ref_audio is not None else None
            if continuation:
                if ref_audio is None:
                    return "Continuation mode requires Speaker Reference Audio.", None
                audio_out, sr = pipe.tts_continuation(
                    target_text=text,
                    ref_audio=speaker_path,
                    ref_text=ref_text.strip() if ref_text and ref_text.strip() else None,
                )
            else:
                audio_out, sr = pipe.tts(text, speaker_audio=speaker_path)
            return "", tensor_to_gradio_audio(audio_out, sr)

        elif task == "SpeechChat":
            if audio is None:
                return "SpeechChat requires audio input.", None
            path = audio_to_tempfile(audio)
            answer = pipe.speech_chat(path)
            return answer, None

        elif task == "TextQA":
            if not text.strip():
                return "TextQA requires text input.", None
            audio_path = audio_to_tempfile(audio) if audio is not None else None
            response = pipe.textqa(text, audio=audio_path)
            return response, None

        else:
            return f"Unknown task: {task}", None

    except Exception as exc:
        return f"Error: {exc}", None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_interface(pipe: RaonPipeline, model_path: str) -> gr.Blocks:
    """Build the Gradio Blocks interface."""

    def on_generate(
        task: str,
        text: str,
        audio: tuple[int, np.ndarray] | None,
        ref_audio: tuple[int, np.ndarray] | None,
        continuation: bool,
        ref_text: str,
    ):
        return run_inference(pipe, task, text, audio, ref_audio, continuation, ref_text)

    def on_task_change(task: str, continuation: bool):
        show_text_in = task in ("TTS", "TextQA")
        show_audio_in = task in ("STT", "TextQA", "SpeechChat")
        show_ref = task == "TTS"
        show_text_out = task != "TTS"
        show_audio_out = task == "TTS"
        continuation_value = bool(continuation) if show_ref else False
        show_ref_text = show_ref and continuation_value
        return (
            gr.update(visible=show_text_in),
            gr.update(visible=show_audio_in),
            gr.update(visible=show_ref),
            gr.update(visible=show_text_out),
            gr.update(visible=show_audio_out),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(visible=show_ref, value=continuation_value),  # continuation checkbox — show/hide with TTS
            gr.update(visible=show_ref_text),
        )

    def on_continuation_change(continuation: bool):
        return gr.update(visible=bool(continuation))

    with gr.Blocks(title="RAON Demo") as demo:
        gr.Markdown(f"# RAON Gradio Demo\n**Model:** `{model_path}`")

        task_dropdown = gr.Dropdown(
            choices=["STT", "TTS", "SpeechChat", "TextQA"],
            value="TextQA",
            label="Task",
        )
        with gr.Row():
            with gr.Column(visible=True) as text_in_col:
                text_input = gr.Textbox(
                    label="Text Input",
                    placeholder="Enter text here...",
                    lines=3,
                )
            with gr.Column(visible=True) as audio_in_col:
                audio_input = gr.Audio(
                    label="Audio Input (optional for TextQA)",
                    type="numpy",
                    sources=["upload", "microphone"],
                )

        with gr.Column(visible=False) as ref_audio_col:
            ref_audio_input = gr.Audio(
                label="Speaker Reference Audio (TTS only)",
                type="numpy",
                sources=["upload"],
            )
            continuation_checkbox = gr.Checkbox(label="Continuation Mode", value=False)
            ref_text_input = gr.Textbox(
                label="Reference Text (optional, auto-transcribed if empty)",
                placeholder="Transcription of the reference audio...",
                lines=2,
                visible=False,
            )

        generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(visible=True) as text_out_col:
                text_output = gr.Textbox(label="Text Output", lines=6)
            with gr.Column(visible=False) as audio_out_col:
                audio_output = gr.Audio(label="Audio Output", interactive=False)

        continuation_checkbox.input(
            fn=on_continuation_change,
            inputs=[continuation_checkbox],
            outputs=[ref_text_input],
            queue=False,
        )
        continuation_checkbox.change(
            fn=on_continuation_change,
            inputs=[continuation_checkbox],
            outputs=[ref_text_input],
            queue=False,
        )

        task_dropdown.change(
            fn=on_task_change,
            inputs=[task_dropdown, continuation_checkbox],
            outputs=[
                text_in_col,
                audio_in_col,
                ref_audio_col,
                text_out_col,
                audio_out_col,
                text_output,
                audio_output,
                continuation_checkbox,
                ref_text_input,
            ],
            queue=False,
        )

        generate_btn.click(
            fn=on_generate,
            inputs=[
                task_dropdown,
                text_input,
                audio_input,
                ref_audio_input,
                continuation_checkbox,
                ref_text_input,
            ],
            outputs=[text_output, audio_output],
        )

        examples = load_examples()
        if examples:
            exs = gr.Examples(
                examples=examples,
                inputs=[
                    task_dropdown,
                    text_input,
                    audio_input,
                    ref_audio_input,
                    ref_text_input,
                ],
                label="Sample Inputs (data/speechllm/eval)",
            )
            exs.dataset.click(
                fn=on_task_change,
                inputs=[task_dropdown, continuation_checkbox],
                outputs=[
                    text_in_col,
                    audio_in_col,
                    ref_audio_col,
                    text_out_col,
                    audio_out_col,
                    text_output,
                    audio_output,
                    continuation_checkbox,
                    ref_text_input,
                ],
                queue=False,
            )

        demo.queue()
    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAON Gradio demo (local model)")
    parser.add_argument(
        "--model-path",
        default="KRAFTON/Raon-Speech-9B",
        help="Local model path or HuggingFace model ID (default: KRAFTON/Raon-Speech-9B)",
    )
    parser.add_argument("--config", default=None, help="Path to infer.yaml-style task defaults")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--dtype", default="bfloat16", help="Data type (default: bfloat16)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for private models")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio server host")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    return parser.parse_args()


def _ensure_hub_code(model_path: str, hub_repo: str = "KRAFTON/Raon-Speech-9B") -> None:
    """If model_path is a local directory without Hub code files, download them from Hub."""
    from huggingface_hub import hf_hub_download

    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return  # Hub ID, not local path — nothing to do
    for fname in ("configuration_raon.py", "modeling_raon.py"):
        if not (model_dir / fname).exists():
            print(f"  Downloading {fname} from {hub_repo}...")
            hf_hub_download(hub_repo, fname, local_dir=str(model_dir))


def main() -> None:
    args = parse_args()

    AutoConfig.register("raon", RaonConfig)
    AutoModel.register(RaonConfig, RaonModel)

    if args.hf_token:
        login(token=args.hf_token)

    _ensure_hub_code(args.model_path)

    print(f"Loading RaonPipeline from: {args.model_path} ...")
    pipe = RaonPipeline(args.model_path, device=args.device, dtype=args.dtype, config=args.config)
    print("Pipeline ready.\n")

    demo = build_interface(pipe, args.model_path)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
