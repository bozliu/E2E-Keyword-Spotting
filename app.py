from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
src_str = str(SRC)
if src_str in sys.path:
    sys.path.remove(src_str)
sys.path.insert(0, src_str)

from kws.demo.web import create_gradio_app


def build_app():
    return create_gradio_app(checkpoint="auto", selection_profile="stable", sensitivity_profile="strict")


app = build_app()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
