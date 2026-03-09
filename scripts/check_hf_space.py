from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or wait on a Hugging Face Space runtime.")
    parser.add_argument(
        "--space-id",
        default=os.environ.get("HF_SPACE_ID"),
        help="Hugging Face Space id, e.g. user/e2e-keyword-spotting-demo. Falls back to HF_SPACE_ID.",
    )
    parser.add_argument("--wait-ready", action="store_true", help="Poll until the Space reaches RUNNING.")
    parser.add_argument(
        "--require-sha-match",
        action="store_true",
        help="When waiting, also require runtime sha to match repo sha before succeeding.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Maximum seconds to wait when --wait-ready is enabled.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=10,
        help="Polling interval in seconds when --wait-ready is enabled.",
    )
    args = parser.parse_args()
    if not args.space_id:
        parser.error("--space-id is required unless HF_SPACE_ID is set")
    return args


def _extract_runtime_sha(raw: dict[str, Any]) -> str | None:
    replicas = raw.get("replicas")
    if isinstance(replicas, list):
        for replica in replicas:
            if isinstance(replica, dict) and replica.get("sha"):
                return str(replica["sha"])
    if isinstance(replicas, dict) and replicas.get("sha"):
        return str(replicas["sha"])
    if raw.get("sha"):
        return str(raw["sha"])
    return None


def fetch_space_status(api: HfApi, space_id: str) -> dict[str, Any]:
    info = api.repo_info(repo_id=space_id, repo_type="space")
    runtime = api.get_space_runtime(repo_id=space_id)
    raw = getattr(runtime, "raw", {}) or {}
    return {
        "space_id": space_id,
        "private": getattr(info, "private", None),
        "repo_sha": getattr(info, "sha", None),
        "runtime_sha": _extract_runtime_sha(raw),
        "stage": getattr(runtime, "stage", None),
        "host": raw.get("host"),
        "subdomain": raw.get("subdomain"),
        "domains": raw.get("domains"),
        "raw": raw,
    }


def _is_ready(status: dict[str, Any], require_sha_match: bool) -> bool:
    if status.get("stage") != "RUNNING":
        return False
    if require_sha_match:
        return bool(status.get("runtime_sha")) and status.get("runtime_sha") == status.get("repo_sha")
    return True


def main() -> None:
    args = parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    deadline = time.time() + args.timeout_seconds
    attempt = 0

    while True:
        attempt += 1
        status = fetch_space_status(api, args.space_id)
        status["attempt"] = attempt
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str), flush=True)

        if not args.wait_ready:
            return
        if _is_ready(status, args.require_sha_match):
            return
        if time.time() >= deadline:
            raise SystemExit(
                f"Timed out waiting for Space '{args.space_id}' to become ready"
                f"{' with matching runtime sha' if args.require_sha_match else ''}."
            )
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
