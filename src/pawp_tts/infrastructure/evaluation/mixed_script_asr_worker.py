"""Stdin/stdout worker for a local mixed-script ASR checkpoint.

Runs inside the model's own isolated Python environment (never imported by the main
package). Protocol: one JSON object per line in (`{"path", "lang", "mode"}`), one JSON
object per line out (`{"text"}` or `{"error"}`).
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    model_dir = sys.argv[1]
    sys.path.insert(0, model_dir)
    from indic_transcribe import IndicTranscribe  # type: ignore[import-not-found]

    model = IndicTranscribe.from_pretrained(model_dir, device="cuda")
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            out = model(req["path"], lang=req.get("lang", "ne"), mode=req.get("mode", "mixed"))
            text = out if isinstance(out, str) else str(getattr(out, "text", out))
            print(json.dumps({"text": text}, ensure_ascii=False), flush=True)
        except Exception as exc:  # report per-request failures, keep serving
            print(json.dumps({"error": str(exc)[:300]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
