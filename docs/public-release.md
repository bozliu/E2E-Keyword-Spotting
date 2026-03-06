# Public Release Workflow

## Branch model
- `main`: latest stable public release
- `legacy-v1`: preserved thesis-era public version
- `codex/update-final-public-release`: release-prep branch used for cleanup and review

## Suggested release flow

```bash
git checkout main
git pull origin main
git checkout -b legacy-v1 origin/main
git push origin legacy-v1
git tag -a v1.0-legacy origin/main -m "Legacy public version"
git push origin v1.0-legacy

git checkout -b codex/update-final-public-release origin/main
# copy cleaned v2 tree, run checks, commit in logical chunks

git push origin codex/update-final-public-release
# open PR into main
# merge after CI passes

git checkout main
git pull origin main
git tag -a v2.0-public -m "Public v2 release"
git push origin v2.0-public
```

## Pull request checklist
- no tracked dataset audio
- no checkpoints in Git history for the release branch
- no local absolute paths or personal identifiers in tracked files
- README updated with public instructions, figures, and benchmark tables
- CI passes
- release assets uploaded to GitHub Releases

## CI scope
The GitHub Actions workflow in this repository is intentionally lightweight:
- install dependencies
- import the package
- run focused tests
- validate the public helper scripts

Full training is intentionally excluded from CI.
