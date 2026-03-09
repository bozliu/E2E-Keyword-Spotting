# Public Release Workflow

## Branch model
- `main`: latest stable public release
- `legacy-v1`: preserved thesis-era public version
- `v2`: preserved read-only branch for the former recent-main line

## Suggested release flow

```bash
git checkout main
git pull origin main
git checkout -b legacy-v1 origin/main
git push origin legacy-v1
git tag -a v1.0-legacy origin/main -m "Legacy public version"
git push origin v1.0-legacy

git tag -a v2.0.0 origin/main -m "Public v2 snapshot before v3"
git push origin v2.0.0
git branch v2 v2.0.0
git push origin v2

git checkout main
git pull origin main
git tag -a v3.0.0 -m "Public v3 release"
git push origin v3.0.0
```

## Pull request checklist
- no tracked dataset audio
- no checkpoints in Git history for the release branch
- no local absolute paths or personal identifiers in tracked files
- README updated with public instructions, figures, and benchmark tables
- README and release docs describe `legacy-v1`, `v2`, and `main` consistently
- CI passes
- release assets uploaded to GitHub Releases

## CI scope
The GitHub Actions workflow in this repository is intentionally lightweight:
- install dependencies
- import the package
- run focused tests
- validate the public helper scripts

Full training is intentionally excluded from CI.
