"""Local HTTP bridge for an external voice/LiveKit agent.

Run this on the GPU computer.  POST a WAV/MP3/M4A file to /analyze and receive
the model's natural-language answer plus conservative environment tags.
"""

import argparse
import tempfile
import threading
from pathlib import Path

from flask import Flask, jsonify, request

from infer_offline import InferenceEngine, get_best_device, parse_environment
from src.audiointeraction.generate.base import streaming_generate


def create_app(checkpoint_dir: str) -> Flask:
    app = Flask(__name__)
    inference_lock = threading.Lock()  # GPU model/cache is single-session.
    print("[boot] loading AudioInteraction weights once; this can take a few minutes")
    engine = InferenceEngine(checkpoint_dir=checkpoint_dir, device=get_best_device())

    @app.get("/health")
    def health():
        return {"ok": True, "device": str(get_best_device())}

    @app.post("/analyze")
    def analyze():
        audio = request.files.get("audio")
        if audio is None or not audio.filename:
            return jsonify(error="Send multipart/form-data with an `audio` file."), 400
        suffix = Path(audio.filename).suffix.lower() or ".wav"
        if suffix not in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}:
            return jsonify(error="Unsupported audio extension."), 415
        with tempfile.TemporaryDirectory(prefix="audiointeraction-") as tmp:
            path = Path(tmp) / f"input{suffix}"
            audio.save(path)
            with inference_lock:
                engine.run(audio_paths=[str(path)])
                reply_text = getattr(streaming_generate, "last_reply_text", "")
        return jsonify(reply_text=reply_text, environment=parse_environment(reply_text))

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    create_app(args.checkpoint_dir).run(host=args.host, port=args.port, threaded=True)
