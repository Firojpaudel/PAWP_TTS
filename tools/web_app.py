"""Studio-grade interactive web app for PAWP-TTS with True Chunked Streaming and Live Metrics.

Zero external web framework dependencies (Python standard library http.server + HTML5 Canvas + Web Audio API).
"""

from __future__ import annotations

import io
import json
import os
import struct
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import torch

from pawp_tts.domain.entities.models import LanguageCode
from pawp_tts.infrastructure.evaluation.benchmark_suite import _synthesize_utterance
from pawp_tts.infrastructure.inference.streaming import PyTorchStreamingRunner
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.training.trainer import CheckpointManager
from pawp_tts.interfaces.container import build_container

print("Initializing PAWP-TTS engine...")
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

streaming_runner = PyTorchStreamingRunner(model, device=str(DEVICE))

# Warm up both full and streaming graphs
try:
    _w_seq = phonemize_usecase.execute("test")
    _w_p_ids = torch.tensor([[t.token_id for t in _w_seq.tokens]], dtype=torch.long, device=DEVICE)
    _w_l_ids = torch.zeros_like(_w_p_ids)
    _w_s_ids = torch.zeros_like(_w_p_ids)
    _w_b_flags = torch.zeros((1, _w_p_ids.shape[1], 4), dtype=torch.float32, device=DEVICE)
    _w_l_masks = torch.zeros((1, _w_p_ids.shape[1], 2), dtype=torch.float32, device=DEVICE)
    for _ in streaming_runner.stream_pcm(
        _w_p_ids, _w_l_ids, _w_s_ids, _w_b_flags, _w_l_masks, first_chunk_frames=6, chunk_frames=16
    ):
        pass
    print("Streaming engine warmed up.")
except Exception as e:
    print(f"Warmup warning: {e}")

print(f"PAWP-TTS engine active on {DEVICE} (speaker_id={SPEAKER_ID}).")


def _prepare_tensors(text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    seq = phonemize_usecase.execute(text)
    tokens = seq.tokens
    lang_map = {LanguageCode.SHARED: 0, LanguageCode.EN: 1, LanguageCode.NE: 2, LanguageCode.MIXED: 3}
    p_ids = torch.tensor([[t.token_id for t in tokens]], dtype=torch.long, device=DEVICE)
    l_ids = torch.tensor([[lang_map.get(t.language, 0) for t in tokens]], dtype=torch.long, device=DEVICE)
    s_ids = torch.tensor([[min(t.stress, 3) for t in tokens]], dtype=torch.long, device=DEVICE)
    b_flags = torch.tensor(
        [
            [
                [
                    1.0 if t.is_word_start else 0.0,
                    1.0 if t.is_word_end else 0.0,
                    1.0 if t.is_syllable_start else 0.0,
                    1.0 if t.is_phrase_break else 0.0,
                ]
                for t in tokens
            ]
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    b_size, t_size = p_ids.shape
    l_masks = torch.zeros(b_size, t_size, 2, dtype=torch.float32, device=DEVICE)
    l_masks[l_ids == 1, 0] = 1.0
    l_masks[l_ids == 2, 1] = 1.0
    phoneme_tags = [f"{t.symbol} ({t.language.value})" for t in tokens]
    return p_ids, l_ids, s_ids, b_flags, l_masks, phoneme_tags


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PAWP-TTS · Studio Neural Speech Playground</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card: #111827;
      --card-border: #1f293d;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-glow: rgba(56, 189, 248, 0.15);
      --accent: #34d399;
      --accent-glow: rgba(52, 211, 153, 0.15);
      --danger: #f87171;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 32px 16px;
    }
    .container {
      width: 100%;
      max-width: 820px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 18px;
    }
    .badge-bar { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 500;
      color: #cbd5e1;
    }
    .badge.active {
      background: var(--primary-glow);
      border-color: rgba(56, 189, 248, 0.4);
      color: var(--primary);
    }
    .badge-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
    h1 { font-size: 26px; font-weight: 700; letter-spacing: -0.02em; color: #fff; }
    .card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    textarea {
      width: 100%;
      min-height: 110px;
      background: #0b1120;
      border: 1px solid #223049;
      border-radius: 12px;
      color: var(--text);
      font-family: inherit;
      font-size: 16px;
      line-height: 1.5;
      padding: 16px;
      resize: vertical;
      outline: none;
      transition: all 0.2s ease;
    }
    textarea:focus { border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-glow); }
    .presets { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .preset-btn {
      background: #182234;
      border: 1px solid #28374f;
      color: #94a3b8;
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .preset-btn:hover { background: #223049; color: #f1f5f9; border-color: #38bdf8; }
    .controls {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 18px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .primary-btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
      color: white;
      border: none;
      padding: 12px 24px;
      border-radius: 10px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
      transition: all 0.2s ease;
    }
    .primary-btn:hover { background: linear-gradient(135deg, #0369a1 0%, #075985 100%); transform: translateY(-1px); }
    .primary-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .stream-toggle {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 14px;
      color: #cbd5e1;
      cursor: pointer;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }
    .metric-card {
      background: #0b1120;
      border: 1px solid #1f2c42;
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .metric-label { font-size: 12px; color: var(--text-muted); font-weight: 500; }
    .metric-val { font-size: 20px; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: var(--primary); }
    .metric-val.green { color: var(--accent); }
    .visualizer-card {
      background: #0b1120;
      border: 1px solid #1f2c42;
      border-radius: 14px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      margin-top: 18px;
    }
    canvas {
      width: 100%;
      height: 70px;
      background: #070a12;
      border-radius: 8px;
    }
    .chunks-flow {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      max-height: 80px;
      overflow-y: auto;
      padding: 4px 0;
    }
    .chunk-chip {
      background: #172554;
      border: 1px solid #1e40af;
      color: #93c5fd;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 6px;
      animation: popIn 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .chunk-chip.first {
      background: #064e3b;
      border-color: #059669;
      color: #6ee7b7;
      font-weight: 600;
    }
    @keyframes popIn { 0% { transform: scale(0.6); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }
    .phonemes-box {
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: #94a3b8;
      background: #070b14;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 10px 14px;
      max-height: 90px;
      overflow-y: auto;
      word-break: break-all;
      margin-top: 8px;
    }
    audio { width: 100%; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>PAWP-TTS Studio</h1>
        <div class="badge-bar">
          <span class="badge active"><span class="badge-dot"></span>1.49M Deployed Params</span>
          <span class="badge">iSTFT ConvNeXt 24 kHz</span>
          <span class="badge">105k Nepali Lexicon</span>
          <span class="badge">True Chunked Streaming</span>
        </div>
      </div>
      <div id="engineStatus" class="badge active" style="margin-top: 4px;">Engine Ready</div>
    </header>

    <div class="card">
      <textarea id="textInput" placeholder="यहाँ नेपाली वा English वा दुवै mixed text लेख्नुहोस्...">नमस्ते, PAWP-TTS मा तपाईंलाई स्वागत छ! आजको मौसम निकै रमाइलो छ।</textarea>

      <div class="presets">
        <button class="preset-btn" onclick="setPreset('नमस्ते, PAWP-TTS मा तपाईंलाई स्वागत छ! आजको मौसम निकै रमाइलो छ।')">🇳🇵 नमस्ते (Nepali)</button>
        <button class="preset-btn" onclick="setPreset('Hello and welcome to PAWP bilingual ultra-compact neural speech synthesis.')">🇬🇧 Introduction (English)</button>
        <button class="preset-btn" onclick="setPreset('आज मेरो office मा एउटा urgent project meeting छ, please join timely.')">🔀 Code-Mixed</button>
        <button class="preset-btn" onclick="setPreset('शहर र सहर दुवै काठमाडौंको मानक उच्चारणमा सहर जस्तै बोलिन्छ।')">🎯 Sibilant Merger (स/श/ष)</button>
        <button class="preset-btn" onclick="setPreset('पाता र पत्ता, पना र पन्ना मा फरक स्पष्ट सुनिन्छ।')">⚡ Gemination (पत्ता/पाता)</button>
      </div>

      <div class="controls">
        <div style="display: flex; gap: 14px; align-items: center;">
          <button id="playBtn" class="primary-btn" onclick="handleSynthesize()">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
            Stream Speech
          </button>
        </div>

        <label class="stream-toggle">
          <input type="checkbox" id="streamingMode" checked>
          <span>True Chunked Streaming (Low-Latency TTFA)</span>
        </label>
      </div>

      <div class="metrics-grid">
        <div class="metric-card">
          <span class="metric-label">TIME TO FIRST AUDIO (TTFA)</span>
          <span id="metricTTFA" class="metric-val green">-- ms</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">STREAMING CHUNKS</span>
          <span id="metricChunks" class="metric-val">--</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">AUDIO DURATION</span>
          <span id="metricDur" class="metric-val">-- s</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">SYNTHESIS RTF</span>
          <span id="metricRTF" class="metric-val green">--x</span>
        </div>
      </div>

      <div class="visualizer-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="font-size: 13px; font-weight: 600; color: #cbd5e1;">Live Audio Waveform</span>
          <span id="statusTxt" style="font-size: 12px; color: var(--primary);">Idle</span>
        </div>
        <canvas id="vizCanvas"></canvas>

        <div style="margin-top: 4px;">
          <span style="font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase;">Streaming Buffer Timeline:</span>
          <div id="chunksFlow" class="chunks-flow"></div>
        </div>

        <div id="staticAudioBox" style="display: none; margin-top: 8px;">
          <audio id="staticPlayer" controls></audio>
        </div>
      </div>

      <div style="margin-top: 16px;">
        <span style="font-size: 12px; font-weight: 600; color: var(--text-muted);">PHONEME SEQUENCE & TOKENS:</span>
        <div id="phonemesBox" class="phonemes-box">Type text above to inspect phonemes...</div>
      </div>
    </div>
  </div>

  <script>
    let audioCtx = null;
    let nextStartTime = 0;
    let analyser = null;
    let animFrame = null;

    function initAudio() {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 64;
        analyser.connect(audioCtx.destination);
        startVisualizer();
      }
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }
    }

    function setPreset(txt) {
      document.getElementById('textInput').value = txt;
    }

    function startVisualizer() {
      const canvas = document.getElementById('vizCanvas');
      const ctx = canvas.getContext('2d');
      const bufferLen = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLen);

      function render() {
        animFrame = requestAnimationFrame(render);
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;

        analyser.getByteFrequencyData(dataArray);
        ctx.fillStyle = '#070a12';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLen) * 1.8;
        let x = 0;
        for (let i = 0; i < bufferLen; i++) {
          const v = dataArray[i] / 255.0;
          const barHeight = v * (canvas.height - 8);

          const grad = ctx.createLinearGradient(0, canvas.height, 0, 0);
          grad.addColorStop(0, '#0284c7');
          grad.addColorStop(1, '#34d399');

          ctx.fillStyle = grad;
          ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
          x += barWidth;
        }
      }
      render();
    }

    async function handleSynthesize() {
      const text = document.getElementById('textInput').value.trim();
      if (!text) return;
      initAudio();

      const isStreaming = document.getElementById('streamingMode').checked;
      const btn = document.getElementById('playBtn');
      const statusTxt = document.getElementById('statusTxt');
      const chunksFlow = document.getElementById('chunksFlow');
      const metricTTFA = document.getElementById('metricTTFA');
      const metricChunks = document.getElementById('metricChunks');
      const metricDur = document.getElementById('metricDur');
      const metricRTF = document.getElementById('metricRTF');
      const phonemesBox = document.getElementById('phonemesBox');

      btn.disabled = true;
      chunksFlow.innerHTML = '';
      statusTxt.innerText = 'Connecting & phonemizing...';
      metricTTFA.innerText = '-- ms';
      metricChunks.innerText = '0';
      metricDur.innerText = '-- s';
      metricRTF.innerText = '--';

      const startTime = performance.now();
      let firstChunkTime = null;
      let totalSamples = 0;
      let chunkCount = 0;

      try {
        if (isStreaming) {
          document.getElementById('staticAudioBox').style.display = 'none';
          const response = await fetch('/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
          });

          if (!response.ok) throw new Error("Streaming error: " + response.statusText);
          const reader = response.body.getReader();

          nextStartTime = audioCtx.currentTime + 0.05; // initial small buffer
          statusTxt.innerText = 'Streaming audio...';

          let pendingBytes = new Uint8Array(0);

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            if (firstChunkTime === null) {
              firstChunkTime = performance.now();
              const ttfa = (firstChunkTime - startTime).toFixed(1);
              metricTTFA.innerText = ttfa + ' ms';
            }

            // Combine bytes
            const combined = new Uint8Array(pendingBytes.length + value.length);
            combined.set(pendingBytes);
            combined.set(value, pendingBytes.length);
            pendingBytes = combined;

            // Chunks are sent as: [4 bytes chunk size (int32)][raw float32 PCM samples]
            while (pendingBytes.length >= 4) {
              const view = new DataView(pendingBytes.buffer, pendingBytes.byteOffset, pendingBytes.byteLength);
              const chunkSize = view.getInt32(0, true);

              if (chunkSize <= 0) {
                // Header or metadata marker
                pendingBytes = pendingBytes.slice(4);
                break;
              }

              const totalPacketLen = 4 + chunkSize;
              if (pendingBytes.length < totalPacketLen) break;

              // Extract float32 samples
              const pcmBytes = pendingBytes.slice(4, totalPacketLen);
              const floatSamples = new Float32Array(pcmBytes.buffer, pcmBytes.byteOffset, pcmBytes.byteLength / 4);
              pendingBytes = pendingBytes.slice(totalPacketLen);

              // Schedule playback in AudioContext
              const buffer = audioCtx.createBuffer(1, floatSamples.length, 24000);
              buffer.copyToChannel(floatSamples, 0);

              const src = audioCtx.createBufferSource();
              src.buffer = buffer;
              src.connect(analyser);

              const scheduledTime = Math.max(audioCtx.currentTime, nextStartTime);
              src.start(scheduledTime);
              nextStartTime = scheduledTime + buffer.duration;

              totalSamples += floatSamples.length;
              chunkCount++;

              // Update UI chip
              const chip = document.createElement('span');
              chip.className = 'chunk-chip' + (chunkCount === 1 ? ' first' : '');
              chip.innerText = `#${chunkCount} (${(floatSamples.length / 24).toFixed(0)}ms)`;
              chunksFlow.appendChild(chip);
              metricChunks.innerText = chunkCount;
            }
          }

          const totalDuration = totalSamples / 24000;
          const totalElapsed = (performance.now() - startTime) / 1000;
          metricDur.innerText = totalDuration.toFixed(2) + ' s';
          metricRTF.innerText = (totalElapsed / totalDuration).toFixed(3) + 'x';
          statusTxt.innerText = 'Completed streaming playback';

        } else {
          // Standard full generation
          statusTxt.innerText = 'Generating full waveform...';
          const response = await fetch('/synth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
          });
          const blob = await response.blob();
          const elapsed = (performance.now() - startTime).toFixed(1);
          metricTTFA.innerText = elapsed + ' ms';
          metricChunks.innerText = '1 (full)';

          const url = URL.createObjectURL(blob);
          const staticPlayer = document.getElementById('staticPlayer');
          staticPlayer.src = url;
          document.getElementById('staticAudioBox').style.display = 'block';
          staticPlayer.play();
          statusTxt.innerText = 'Playback ready';
        }

        // Fetch phonemes preview
        fetch('/phonemes', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        }).then(r => r.json()).then(data => {
          phonemesBox.innerText = data.phonemes.join(' · ');
        });

      } catch (err) {
        statusTxt.innerText = 'Error: ' + err.message;
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""


class StudioTTSHandler(BaseHTTPRequestHandler):
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

        if self.path == "/phonemes":
            seq = phonemize_usecase.execute(text)
            phonemes = [f"{t.symbol}" for t in seq.tokens]
            resp = json.dumps({"phonemes": phonemes}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp)

        elif self.path == "/stream":
            p_ids, l_ids, s_ids, b_flags, l_masks, _ = _prepare_tensors(text)

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            try:
                for chunk in streaming_runner.stream_pcm(
                    p_ids, l_ids, s_ids, b_flags, l_masks, first_chunk_frames=6, chunk_frames=16
                ):
                    pcm_data = chunk.astype(np.float32).tobytes()
                    # Packet format: [4 bytes int32 length] + [raw bytes]
                    packet = struct.pack("<I", len(pcm_data)) + pcm_data
                    # HTTP chunked transfer framing
                    chunk_header = f"{len(packet):X}\r\n".encode("ascii")
                    self.wfile.write(chunk_header + packet + b"\r\n")
                    self.wfile.flush()

                # End HTTP chunked transfer
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == "/synth":
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
        pass  # keep console clean


def main() -> None:
    port = 7860
    server = HTTPServer(("0.0.0.0", port), StudioTTSHandler)
    print(f"\n=======================================================")
    print(f"🚀 PAWP-TTS Studio Streaming UI LIVE at: http://localhost:{port}")
    print(f"   · True Chunked Streaming Active (TTFA < 20 ms)")
    print(f"   · 24 kHz Web Audio Buffer & Live Canvas Visualizer")
    print(f"=======================================================\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
