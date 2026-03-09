from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut over a validated diagnostic Hugging Face Space into production.")
    parser.add_argument("--source-space-id", required=True, help="Validated diagnostic Space id to promote.")
    parser.add_argument(
        "--target-space-id",
        default=os.environ.get("HF_SPACE_ID"),
        help="Final public Space id. Falls back to HF_SPACE_ID.",
    )
    parser.add_argument(
        "--delete-target-first",
        action="store_true",
        help="Delete the current target Space before moving the source into its slug.",
    )
    parser.add_argument(
        "--make-public",
        action="store_true",
        help="Set the moved target Space visibility to public after cutover.",
    )
    parser.add_argument("--wait-ready", action="store_true", help="Wait for the final target Space to reach RUNNING.")
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
    if not args.target_space_id:
        parser.error("--target-space-id is required unless HF_SPACE_ID is set")
    if args.source_space_id == args.target_space_id:
        parser.error("--source-space-id and --target-space-id must differ")
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
        "domains": raw.get("domains"),
        "raw": raw,
    }


def wait_until_ready(
    api: HfApi,
    space_id: str,
    *,
    timeout_seconds: int,
    interval_seconds: int,
    require_sha_match: bool,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while True:
        status = fetch_space_status(api, space_id)
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str), flush=True)
        if status.get("stage") == "RUNNING" and (
            not require_sha_match
            or (status.get("runtime_sha") and status.get("runtime_sha") == status.get("repo_sha"))
        ):
            return status
        if time.time() >= deadline:
            raise SystemExit(
                f"Timed out waiting for Space '{space_id}' to become ready"
                f"{' with matching runtime sha' if require_sha_match else ''}."
            )
        time.sleep(interval_seconds)


def main() -> None:
    args = parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    whoami = api.whoami()
    owner = whoami.get("name") or whoami.get("email") or "authenticated-user"
    print(f"Authenticated with Hugging Face as: {owner}")

    source_status = fetch_space_status(api, args.source_space_id)
    print(json.dumps({"source_before_cutover": source_status}, ensure_ascii=False, indent=2, default=str), flush=True)

    if args.delete_target_first:
        print(f"Deleting target Space: {args.target_space_id}", flush=True)
        api.delete_repo(repo_id=args.target_space_id, repo_type="space", missing_ok=True)

    print(f"Moving Space {args.source_space_id} -> {args.target_space_id}", flush=True)
    api.move_repo(from_id=args.source_space_id, to_id=args.target_space_id, repo_type="space")

    if args.make_public:
        print(f"Setting target Space public: {args.target_space_id}", flush=True)
        api.update_repo_settings(repo_id=args.target_space_id, repo_type="space", private=False)

    if args.wait_ready:
        wait_until_ready(
            api,
            args.target_space_id,
            timeout_seconds=args.timeout_seconds,
            interval_seconds=args.interval_seconds,
            require_sha_match=args.require_sha_match,
        )
    else:
        final_status = fetch_space_status(api, args.target_space_id)
        print(json.dumps({"target_after_cutover": final_status}, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
