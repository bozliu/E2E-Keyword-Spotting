# Public Demo Release Checklist

This checklist is for the hosted browser demo only.

Current status:

- the public HF Space is the free CPU/browser demo path
- it must stay clearly separated from the local `accuracy-first` MPS desktop demo
For the broader `v3` repo release flow, see [`docs/public_release_v3.md`](public_release_v3.md).

## Before deployment
- `conda run -n dl python -V` reports Python 3.12
- `conda run -n dl python -m pip install -e . -r requirements-space.txt` succeeds
- `conda run -n dl python -m py_compile app.py scripts/deploy_hf_space.py src/kws/demo/web.py` succeeds
- `conda run --no-capture-output -n dl pytest -q tests/test_demo_web.py tests/test_demo*.py tests/test_keyword_focus*.py` succeeds
- Web smoke inference returns a valid response
- No local/private files are staged for upload: datasets, outputs, cache, user profiles, checkpoints

## Deployment
- `HF_TOKEN` is available locally or via `huggingface-cli login`
- `HF_SPACE_ID` is set correctly or passed to `scripts/release_public_demo.sh`
- `scripts/release_public_demo.sh` completes without errors
- The Space URL is reachable
- For a clean-room validation, `scripts/deploy_hf_space.py --space-id <diag-space> --private` succeeds
- `scripts/check_hf_space.py --space-id <diag-space> --wait-ready --require-sha-match` reports `RUNNING` with matching repo/runtime SHA

## After deployment
- The Space home page renders
- Browser microphone permission works
- Browser microphone recording works
- First inference downloads checkpoint successfully
- README points to the final public URL
- Space README/card says this is a free CPU browser demo, not the same as local `v3`
- No local/private paths appear in the public UI or logs
- If promoting from a private diagnostic Space, `scripts/cutover_hf_space.py --delete-target-first --make-public --wait-ready --require-sha-match` completes before public validation starts
