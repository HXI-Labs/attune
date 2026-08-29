## Summary

- What changed:
- Why:
- Evidence affected:

## Verification

- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `pytest -q`
- [ ] Release gates pass where model/evaluation artifacts changed
- [ ] README/model-card claims match committed reports

## Model and data review

- [ ] No raw audio, credentials, or unapproved third-party weights are included
- [ ] Dataset provenance and split rules are preserved
- [ ] SenseVoiceSmall by FunASR/FunAudioLLM remains attributed
- [ ] Public weight redistribution has an explicit completed review, or the
      Hugging Face repository remains private

## Limitations

- New or changed limitations:
- Follow-up work:
