# AITE demo acceptance runbook

Acceptance is pending, not complete. See [the paper handoff](IEEE_PAPER_HANDOFF.md)
for extraction-runtime, MIME, translation and multiword-output issues. Record
browser/OS/device, code revision, enabled modes and actual observations. A returned
MP4 or a fallback label alone is not a pass for linguistic completeness.

## Start the local demo

```bash
cd ~/Documents/ASL-ISL
AITE_ENABLE_RECOGNITION=false AITE_ENABLE_GEMINI=false AITE_ENABLE_LLM=false \
  .venv/bin/uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser.

## Record results

Use `DEMO_VALIDATION_LOG_TEMPLATE.csv` to capture each test. Preserve a
screenshot or screen recording for every successful end-to-end case.

## Required checks

1. Enter `HELLO WATER`; inspect BOTH meanings. First-word-only output is an
   incomplete-output failure, even if the MP4 plays.
2. Enter `THANK_YOU`; inspect recording and label. Functional playback does not
   establish linguistic approval.
3. After resolving extraction compatibility, restart with recognition enabled.
   Record one supported isolated ASL sign with the webcam; allow browser camera
   permission, then confirm a prediction or a clear low-confidence rejection.
4. Enter an unsupported name such as `NANDITA`; before alphabet assets arrive,
   confirm the labelled fallback. After `alpha_keypoints` arrives, confirm the
   complete N-A-N-D-I-T-A sequence, including repeated letters and transitions.
   Missing letters must not count as successful spelling.
5. Check exact sentences and negated variants for incorrect approximate matches.
   Test denied camera permission, stop/clear, MIME types and unsupported inputs.
6. Test Gemini separately only when intentionally enabled; keep secrets outside
   evidence and record the actual provider/fallback mode and meaning failures.
7. Stop the server with `Ctrl+C` after testing. Leave unexecuted CSV rows blank.

## Scope statement for presentation

This is a research prototype: Gemini output is labelled as an AI-assisted draft,
recorded ISL motion is labelled as playback rather than generation, and unknown
or unsupported input should be labelled, but semantic correctness, complete
multiword output and complete fingerspelling still require acceptance testing.
