## Summary
- 

## Public release checklist
- [ ] No dataset audio is committed
- [ ] No checkpoints are committed to Git history
- [ ] No personal or machine-local paths remain in tracked files
- [ ] README and public docs were updated if behavior changed
- [ ] CI passed
- [ ] Release asset links or checkpoint instructions are still correct

## Validation
- [ ] `pytest -q tests/test_demo*.py tests/test_keyword_focus*.py tests/test_prepare_speech_commands.py`
- [ ] `python -m kws.eval --help`
- [ ] `python -m kws.demo.realtime --help`
