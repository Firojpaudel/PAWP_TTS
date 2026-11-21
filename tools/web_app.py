"""Instant browser UI for temporary PAWP-TTS synthesis without saving files to disk.

Runs with zero external UI dependencies (pure Python http.server + HTML5 Audio).
"""

from __future__ import annotations

import io
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.evaluation.benchmark_suite import _synthesize_utterance
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.training.trainer import CheckpointManager
from pawp_tts.interfaces.container import build_container

print("Loading PAWP-TTS model into memory...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "configs/student_1_5m.yaml"
CKPT_PATH = "artifacts/checkpoints/exp007/latest.safetensors"

cfg = load_student_config(CONFIG_PATH)
model = PAWPBilingualTTS(cfg).to(DEVICE).eval()
CheckpointManager.load(model, CKPT_PATH)

container = build_container()
phonemize_usecase = container.phonemize

target_speaker_name = os.environ.get("PAWP_TARGET_SPEAKER", "target_voice")
SPEAKER_ID = 0
if model.speaker_emb is not None and model.speaker_emb.num_embeddings > 1:
    SPEAKER_ID = cfg.speaker_to_id.get(target_speaker_name, 0) if hasattr(cfg, "speaker_to_id") else 0

print(f"Model ready on {DEVICE} (speaker_id={SPEAKER_ID}).")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PAWP-TTS Interactive Audio Playground</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; background: #0d1117; color: #e6edf3; max-width: 720px; margin: 40px auto; padding: 0 20px; }
    h1 { font-size: 24px; margin-bottom: 8px; color: #58a6ff; }
    p.sub { color: #8b949e; margin-top: 0; margin-bottom: 24px; font-size: 14px; }
    textarea { width: 100%; box-sizing: border-box; background: #161b22; border: 1px solid #30363d; border-radius: 8px; color: #e6edf3; padding: 12px; font-size: 16px; min-height: 100px; resize: vertical; }
    button { background: #238636; color: #fff; border: none; padding: 10px 20px; border-radius: 6px; font-size: 15px; font-weight: 600; cursor: pointer; margin-top: 12px; }
    button:hover { background: #2ea043; }
    button:disabled { background: #234e2c; cursor: wait; }
    .examples { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
    .example-btn { background: #21262d; border: 1px solid #30363d; color: #8b949e; padding: 6px 12px; border-radius: 20px; font-size: 13px; cursor: pointer; }
    .example-btn:hover { background: #30363d; color: #e6edf3; }
    .audio-box { margin-top: 24px; padding: 16px; background: #161b22; border-radius: 8px; border: 1px solid #30363d; display: none; }
    audio { width: 100%; margin-top: 10px; }
    .status { color: #58a6ff; font-size: 14px; margin-top: 10px; }
  </style>
</head>
<body>
  <h1>PAWP-TTS Interactive Audio Playground</h1>
  <p class="sub">1.49M-parameter bilingual neural speech synthesis · Temporary in-memory audio (zero disk clutter)</p>
  
  <textarea id="text" placeholder="Type Nepali, English, or Code-mixed text here...">नमस्ते, PAWP-TTS मा तपाईंलाई स्वागत छ!</textarea>
  
  <div class="examples">
    <button class="example-btn" onclick="setEx('नमस्ते, आज काठमाडौंको मौसम निकै रमाइलो छ।')">🇳🇵 मौसम (Nepali)</button>
    <button class="example-btn" onclick="setEx('Good morning. Your appointment is confirmed for ten thirty tomorrow.')">🇬🇧 Appointment (English)</button>
    <button class="example-btn" onclick="setEx('आज मेरो office मा एउटा important meeting छ।')">🔀 Office Meeting (Code-mix)</button>
    <button class="example-btn" onclick="setEx('हाम्रो नयाँ model ले धेरै राम्रो काम गरिरहेको छ।')">🇳🇵 Model Status</button>
  </div>

  <div style="margin-top: 16px;">
    <button id="synthBtn" onclick="synthesize()">▶ Synthesize & Play</button>
    <span id="status" class="status"></span>
  </div>

  <div id="audioBox" class="audio-box">
    <div style="font-weight: 600; font-size: 14px; color: #7ee787;">✓ Synthesized Audio (24 kHz)</div>
    <audio id="player" controls autoplay></audio>
  </div>

  <script>
    function setEx(str) {
      document.getElementById('text').value = str;
    }

    async function synthesize() {
      const text = document.getElementById('text').value.trim();
      if (!text) return;
      const btn = document.getElementById('synthBtn');
      const status = document.getElementById('status');
      const audioBox = document.getElementById('audioBox');
      const player = document.getElementById('player');

      btn.disabled = true;
      status.innerText = "Synthesizing...";

      try {
        const res = await fetch('/synth', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        if (!res.ok) throw new Error("Synthesis failed");
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        player.src = url;
        audioBox.style.display = "block";
        player.play();
        status.innerText = "";
      } catch (err) {
        status.innerText = "Error: " + err.message;
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""


class SimpleTTSHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/synth":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(body)
                text = str(data.get("text", "")).strip()
            except Exception:
                text = ""

            if not text:
                self.send_response(400)
                self.end_headers()
                return

            _, _, wav, _ = _synthesize_utterance(
                model, phonemize_usecase, text, DEVICE, speaker_id=SPEAKER_ID
            )

            buf = io.BytesIO()
            sf.write(buf, wav, 24000, format="WAV")
            audio_bytes = buf.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio_bytes)))
            self.end_headers()
            self.wfile.write(audio_bytes)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass  # quiet


def run_server(port: int = 7860) -> None:
    server = HTTPServer(("0.0.0.0", port), SimpleTTSHandler)
    print(f"\n=======================================================")
    print(f"🚀 PAWP-TTS Web UI is LIVE at: http://localhost:{port}")
    print(f"   (Open in your browser to test phrases interactively!)")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    run_server(port=7860)
