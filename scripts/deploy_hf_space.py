from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

ROOT = Path(__file__).resolve().parents[1]
SPACE_README = ROOT / "space" / "README.md"
APP_FILE = ROOT / "app.py"
SPACE_REQUIREMENTS = ROOT / "requirements-space.txt"
SRC_DIR = ROOT / "src"
SPACE_BUNDLE_IGNORE_PATTERNS = ("__pycache__", "*.pyc", "*.pyo", "*.egg-info", ".DS_Store")
SPACE_DELETE_PATTERNS = [
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.egg-info",
    "**/*.egg-info/**",
]


def build_space_bundle(bundle_dir: Path) -> None:
    shutil.copy2(SPACE_README, bundle_dir / "README.md")
    shutil.copy2(APP_FILE, bundle_dir / "app.py")
    shutil.copy2(SPACE_REQUIREMENTS, bundle_dir / "requirements.txt")
    shutil.copytree(
        SRC_DIR,
        bundle_dir / "src",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*SPACE_BUNDLE_IGNORE_PATTERNS),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the public browser demo to Hugging Face Spaces.")
    parser.add_argument(
        "--space-id",
        default=os.environ.get("HF_SPACE_ID"),
        help="Hugging Face Space id, e.g. user/e2e-keyword-spotting-demo. Falls back to HF_SPACE_ID.",
    )
    parser.add_argument("--private", action="store_true", help="Create the Space as private")
    parser.add_argument("--skip-create", action="store_true", help="Skip repo creation and only upload files")
    args = parser.parse_args()
    if not args.space_id:
        parser.error("--space-id is required unless HF_SPACE_ID is set")
    return args


def main() -> None:
    args = parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        whoami = api.whoami()
    except Exception as exc:  # pragma: no cover - network/auth specific
        raise SystemExit(
            "Hugging Face authentication is missing. Run 'huggingface-cli login' or set HF_TOKEN before deploying."
        ) from exc

    owner = whoami.get("name") or whoami.get("email") or "authenticated-user"
    print(f"Authenticated with Hugging Face as: {owner}")

    if not args.skip_create:
        try:
            api.create_repo(
                repo_id=args.space_id,
                repo_type="space",
                space_sdk="gradio",
                private=bool(args.private),
                exist_ok=True,
            )
        except HfHubHTTPError as exc:  # pragma: no cover - remote specific
            raise SystemExit(f"Failed to create or access Space '{args.space_id}': {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="kws-space-") as tmp:
        bundle_dir = Path(tmp)
        build_space_bundle(bundle_dir)
        api.upload_folder(
            folder_path=str(bundle_dir),
            repo_id=args.space_id,
            repo_type="space",
            commit_message="Update public browser demo bundle",
            delete_patterns=SPACE_DELETE_PATTERNS,
        )

    print(f"Deployed files to https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
