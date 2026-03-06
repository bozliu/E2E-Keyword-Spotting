#!/usr/bin/env python3
"""Select the default demo checkpoint from existing outputs.

This is a thin wrapper around `python -m kws.demo.rank_checkpoints`.
"""

from __future__ import annotations

from kws.demo.rank_checkpoints import DEFAULT_REPORT_PATH, select_best_checkpoint


def main() -> None:
    chosen, runtime_device = select_best_checkpoint(outputs_root="outputs", device="auto", use_cache=False, rebuild=True)
    print(f"Chosen checkpoint: {chosen} on {runtime_device}")
    print(f"Saved ranking report: {DEFAULT_REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
