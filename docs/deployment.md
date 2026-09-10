# Deployment and defense handoff

See [the current paper handoff](IEEE_PAPER_HANDOFF.md) for known runtime blockers.
Commands are reproduction instructions, not proof of live acceptance. Prefer
binding to `127.0.0.1` for a local-only demo.

## Local run

```bash
.venv/bin/uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. The service uses deterministic local translation
rules by default; set `AITE_ENABLE_LLM=true` only when an approved model is
available and its output has been evaluated.

Gemini refinement is opt-in. Store `GEMINI_API_KEY` outside the repository,
load it in the terminal that launches the service, and set
`AITE_ENABLE_GEMINI=true`. Gemini produces a vocabulary-checked draft, without
semantic/completeness guarantees. Errors normally fall back, potentially through
Llama when also enabled. See [TRANSLATION.md](TRANSLATION.md). This integration
sends text, not the uploaded video, to Gemini.

The demo defaults to `AITE_ENABLE_RECOGNITION=false`. The selected isolated-sign
checkpoint is `models/msasl100_pose_balanced_large_recognition/best_model.pt`;
it passed the predeclared validation and held-out selective-accuracy gate. Upload
extraction currently uses the API Python executable in a subprocess. The older
verification helper uses `.venv-cslrt`, but the upload handler does not; the
MediaPipe compatibility workaround must be reconciled and tested before
claiming the current HTTP upload path works.

To enable the isolated-sign file-upload and webcam path for acceptance testing, run:

```bash
AITE_ENABLE_RECOGNITION=true MPLCONFIGDIR=/tmp/aite-mpl \
  .venv/bin/uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

The supported scope is one MSASL-100 isolated sign per clip, not continuous ASL
sentences. The API applies a 50 MB upload limit, landmark coverage checks, and a
validation-selected confidence threshold. Low-confidence clips return `UNKNOWN`.
Generated avatar videos are stored in `/tmp/aite-generated` and expired files
are removed whenever a new avatar is generated. The default retention period is
one hour; set `AITE_GENERATED_VIDEO_TTL_SECONDS` to change it, or to a negative
value only when an external cleanup policy is in place.

## Docker

```bash
docker build -t aite:local .
docker run --rm -p 8000:8000 aite:local
```

Then open `http://localhost:8000/health` and the UI at `http://localhost:8000`.

The image includes the source, static UI, configs, and recognition checkpoint. It intentionally excludes the training dataset and local virtual environments.

## Verification before a demo

Run these checks from the repository root after installing dependencies:

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/pytest -q
.venv-cslrt/bin/python scripts/validate_msasl_landmarks.py \
  --keypoints-dir data/asl/msasl100_keypoints
.venv/bin/python scripts/evaluate_cslrt_recognition.py \
  --checkpoint models/msasl100_pose_balanced_large_recognition/best_model.pt \
  --keypoints-dir data/asl/msasl100_keypoints \
  --splits-dir data/asl/msasl100_splits \
  --split test \
  --calibration models/msasl100_pose_balanced_large_recognition/calibration.json \
  --output models/msasl100_pose_balanced_large_recognition/test_results.json

# Verify the three implemented video-to-recorded-pose demo paths.
AITE_ENABLE_RECOGNITION=true MPLCONFIGDIR=/tmp/aite-mpl \
  .venv/bin/python scripts/verify_video_to_avatar_demo.py
```

`/generate-avatar` returns a relative `video_url` under `/generated/`; the browser
UI consumes that URL directly. It retrieves recorded landmark motion where a
lookup match exists and otherwise renders an illustrative 2D MP4. Lookup is not
linguistic approval; multiword output may contain only one sign. Neither mode is
a trained or linguistically validated ISL generator.

For an exact match to one of the 97 extracted CSLRT English sentence labels,
the UI additionally retrieves the recorded landmark sequence with the best
hand-tracking coverage and plays it as a labelled 2D pose animation. For
example, enter `He is going into the room` in the manual text field. The API
returns `mode: recorded_sentence_landmark_playback` plus the source sentence
and signer. This is motion retrieval from the dataset—not translation,
recognition, or a trained pose-generation model—and unmatched text continues
to use the word-level recording or illustrative fallback.

## Current validation status

The old MSASL checkpoint reported 1.14% Top-1 and 4.28% Top-5, but the subsequent
audit found that every legacy cached feature array was all-zero. Treat that run
only as evidence of a failed preprocessing pipeline, not as a scientific model
result. The replacement MSASL-100 experiment uses 33 pose and 42 hand landmarks,
official signer-independent splits, quality filtering, validation-only rejection
calibration, and held-out Top-1/Top-5/macro-F1/ECE reporting. Keep recognition
disabled until those new result files exist and meet the declared target.

## Reproduce the experimental 126-class expansion

This experiment is retained for research and mapping triage only. It is not the
API checkpoint and must not be enabled for avatar playback without individual
ISL linguistic review. It combines the validated 100-class MSASL data with
26 exact-English-label INCLUDE-50 candidates; `break` and `elephant` were
excluded because quality filtering left them absent from a held-out split.

```bash
# Train in safe CPU-sized chunks. Repeat the second command until it reports
# completion; every chunk restores model, optimizer, scheduler, and RNG state.
MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python scripts/train_cslrt_recognition.py \
  --config configs/msasl126_include50_candidate_recognition.yaml \
  --max-epochs-per-run 3

MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python scripts/train_cslrt_recognition.py \
  --config configs/msasl126_include50_candidate_recognition.yaml \
  --max-epochs-per-run 3 --resume

MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python scripts/evaluate_cslrt_recognition.py \
  --checkpoint models/msasl126_include50_candidate_recognition/best_model.pt \
  --keypoints-dir data/asl/msasl100_keypoints --splits-dir data/asl/msasl126_splits \
  --split val --output models/msasl126_include50_candidate_recognition/val_results.json

.venv/bin/python scripts/calibrate_msasl_recognition.py \
  --validation-results models/msasl126_include50_candidate_recognition/val_results.json \
  --output models/msasl126_include50_candidate_recognition/calibration.json

MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python scripts/evaluate_cslrt_recognition.py \
  --checkpoint models/msasl126_include50_candidate_recognition/best_model.pt \
  --keypoints-dir data/asl/msasl100_keypoints --splits-dir data/asl/msasl126_splits \
  --split test --calibration models/msasl126_include50_candidate_recognition/calibration.json \
  --output models/msasl126_include50_candidate_recognition/test_results.json

.venv/bin/python scripts/report_candidate_class_metrics.py \
  --results models/msasl126_include50_candidate_recognition/test_results.json \
  --candidates configs/msasl_include50_candidate_glosses.json \
  --output docs/msasl126_candidate_class_metrics.json
```

## Next generation milestone: pose-generation training

The next planned major phase is a pose-generation model trained on normalized
ISL motion sequences. Before training, extract and inspect landmark data from
licensed ISL videos:

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python scripts/extract_isl_landmarks.py \
  --input-dir data/isl/include-50 \
  --output-dir data/isl/keypoints
```

### Completed prerequisite: motion reconstruction baseline

The repository now includes an evaluated temporal motion autoencoder. It learns
to reconstruct observed CSLRT pose sequences and is deliberately **not** a
text-conditioned generator or an avatar model. Run it in CPU-safe chunks:

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python scripts/train_motion_autoencoder.py \
  --config configs/isl_motion_autoencoder.yaml --max-epochs-per-run 10

MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python scripts/train_motion_autoencoder.py \
  --config configs/isl_motion_autoencoder.yaml --max-epochs-per-run 10 --resume
```

The completed baseline used signer 1--5 for training, signer 6 for validation,
and signer 7 only for final testing. Its held-out reconstruction MAE was 0.4034;
this generalization gap is evidence that more diverse, reviewed ISL data and
text/gloss supervision are needed before attempting conditional generation.

### ISL-CSLRT sentence corpus

The sentence-level ISL-CSLRT release stores each recording as an ordered folder
of JPG frames, not a video. Extract its pose and hand landmarks with the
dedicated resumable script:

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv-cslrt/bin/python scripts/extract_cslrt_landmarks.py \
  --input-dir data/isl/cslrt_raw/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Sentence_Level \
  --output-dir data/isl/cslrt_keypoints
```

It safely reuses sequences already written, so rerun the same command after
an interruption. Its `metadata.json` maps each signer sequence to its English
sentence label. These landmarks are training data; they do not make the demo
avatar sign new sentences yet. This project machine requires Python 3.12 and
MediaPipe 0.10.21 in `.venv-cslrt`; the main Python 3.14 environment cannot
reliably initialize the native landmark runtime.

After any landmark schema or normalization change, regenerate rather than
reusing old outputs, then validate them:

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv-cslrt/bin/python scripts/extract_cslrt_landmarks.py \
  --input-dir data/isl/cslrt_raw/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Sentence_Level \
  --output-dir data/isl/cslrt_keypoints --overwrite
.venv/bin/python scripts/validate_cslrt_landmarks.py
```

After both landmark passes finish, create signer-independent manifests:

```bash
.venv/bin/python scripts/build_cslrt_splits.py
```

The checked split assigns signers 1–5 to training, signer 6 to validation,
and signer 7 to testing. Current corpus statistics are 474/96/93 samples and
97/96/93 represented sentence classes respectively. The missing validation
and test classes are recorded in `data/isl/cslrt_splits/summary.json`; they
reflect missing signer/class combinations in the downloaded corpus, rather
than samples being dropped by preprocessing.

Train the sentence-classification baseline from the pre-extracted landmarks:

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python scripts/train_cslrt_recognition.py
```

The loader resamples every sequence to 32 frames and uses all 75 landmarks
(33 pose, 21 left-hand, 21 right-hand). The current v3 configuration uses a
three-layer transformer plus mild coordinate noise, scale jitter, landmark
masking, frame dropout, label smoothing, and a fixed seed. Checkpoints,
vocabulary, and the CSV training log are written to
`models/cslrt_recognition_v3/`, preserving historical experiments. For a disposable
smoke run, use `--epochs 1 --limit 8 --checkpoint-dir /tmp/cslrt-smoke`.

Evaluate only the saved best checkpoint on the held-out signer-7 split:

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python scripts/evaluate_cslrt_recognition.py
```

For the v3 checkpoint, pass
`--checkpoint models/cslrt_recognition_v3/best_model.pt --output
models/cslrt_recognition_v3/test_results.json`.

The original v1/v2 runs used mixed coordinate systems because a separate hand
pass overwrote shoulder-normalized hand landmarks with raw image coordinates.
Their reported scores (including v1's 9.68% test top-1) are retained only as
historical diagnostics and are not valid model-quality benchmarks. Retrain
only after the schema-v2 validator passes.

### CSLRT isolated-word auxiliary data

The corpus also contains 1,036 labelled still images across 114 ISL words.
They are not sentence clips, so keep their training separate from the
sentence recognizer. Extract schema-v2 landmarks with:

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv-cslrt/bin/python scripts/extract_cslrt_word_landmarks.py \
  --input-dir data/isl/cslrt_raw/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level \
  --output-dir data/isl/cslrt_word_keypoints
```

Build deterministic class-stratified image splits and train the isolated-word
baseline (with class-balanced sampling):

```bash
.venv/bin/python scripts/build_cslrt_word_splits.py
MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python scripts/train_cslrt_recognition.py \
  --config configs/cslrt_word_recognition.yaml
```

Evaluate it with `--keypoints-dir data/isl/cslrt_word_keypoints --splits-dir
data/isl/cslrt_word_splits --checkpoint models/cslrt_word_recognition/best_model.pt`.

The completed baseline scored **7.69% top-1** and **31.36% top-5** on 169
held-out still images across 114 labels. It is an auxiliary experiment only:
the source is single images rather than signing videos, the classes are highly
uneven, and it must not be connected to the app's uploaded-video recognition
path. Its detailed result is saved at
`models/cslrt_word_recognition/test_results.json`.

## Landmark playback mapping

`configs/avatar_gloss_map.json` maps translator aliases to the 61 available
recorded landmark classes. It is deliberately marked
`draft_pending_isl_expert_review`: direct dataset labels are useful for testing
playback, but an ISL signer must validate language-specific aliases before a
production or research claim. Validate map coverage after regenerating data:

```bash
.venv/bin/python scripts/validate_avatar_gloss_map.py
```

For translation evaluation, provide a reviewed held-out JSON array with either
`asl_gloss`/`isl_gloss` or `asl`/`isl` fields. The evaluator makes genuine
translator predictions and refuses to invent a fallback test set:

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python scripts/eval_translation.py \
  --test-set data/translation_pairs_test.json \
  --output models/translation/eval_results.json
```

## Hosting handoff

Deploy the repository on Render (Docker runtime) or Hugging Face Spaces (Docker SDK), expose port `8000`, and set `AITE_ENABLE_LLM=false`. A public deploy needs a project owner to authenticate and approve publication; do not expose a public endpoint until the team has reviewed its model limitations.

## Defense recording outline

Record a 2–3 minute browser walkthrough after deployment:

1. Show the health indicator and describe **Recognition**; upload one MS-ASL clip in Media workspace.
2. Show the predicted ASL gloss and describe **Translation**; point out the ISL gloss output.
3. Show the playable, labelled **illustrative 2D avatar** generated from the ISL gloss.

Use this exact limitation in narration/captions: “The avatar is an illustrative 2D demo renderer, not a validated linguistic signing synthesizer.”

## Conference submission checklist

There is no `docs/main.pdf` in this repository, so an IEEE automated-format check cannot yet be performed. Before a PI submits, verify the final PDF with the target conference’s checker, then have the authorized corresponding author enter the final author names, emails, affiliations, abstract, and declarations on the chosen portal.
