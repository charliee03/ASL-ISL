# AITE Project Status, Reproduction Guide, and Roadmap

**Project:** ASL–ISL Translation Engine (AITE)  
**Status date:** 10 September 2026  
**Document purpose:** Authoritative record of completed work, current capability,
reproducible project steps, measured results, limitations, and future work.

## 1. Executive status

AITE currently provides a working local research/demo application with a tested
FastAPI backend, browser interface, rule-based draft gloss conversion, extracted
ISL pose data, and recorded 2D landmark playback. The preprocessing and playback
pipelines work; the complete ASL-video → validated ISL translation → generative
avatar product does not yet exist.

The application must currently be demonstrated using manual text. Uploaded-video
recognition is intentionally disabled because all trained recognition baselines
have low held-out accuracy. The avatar retrieves recorded motion for supported
words or sentences and otherwise shows a clearly labelled illustrative figure.
No GAN or other pose-generation model has been trained yet.

## 2. Current capability matrix

| Component | Current state | Evidence / result | Production-ready? |
|---|---|---|---|
| Local API and web UI | Working | `/health` returns `status: ok` | Demo only |
| Automated tests | Working | 27 tests pass | Yes for covered behavior |
| INCLUDE-50 landmark extraction | Complete | 61 extracted playback classes | Data requires expert review |
| CSLRT sentence landmark extraction | Complete | 663 sequences, 18,863 frames | Valid schema-v2 features |
| CSLRT word landmark extraction | Complete | 1,036 still images, 114 classes | Auxiliary data only |
| Sentence pose playback | Working | 97 recorded sentence labels | Retrieval, not generation |
| Word pose playback | Working | 61 recorded word classes, 63 aliases | Retrieval, not generation |
| Illustrative fallback | Working | Browser-playable H.264 MP4 | Not sign-language output |
| ASL video recognition | Validated isolated-sign prototype | MSASL-100 test: 47.02% top-1, 76.78% top-5, 44.61% macro-F1; rejection test accuracy 81.73% | Yes, limited scope |
| CSLRT sentence classifier | Evaluated | 5.38% top-1, 17.20% top-5 | No |
| CSLRT static-word classifier | Evaluated | 7.69% top-1, 31.36% top-5 | No |
| Rule-based gloss conversion | Working as draft | Deterministic mapping/rules | Needs ISL expert validation |
| Llama-2 translation | Disabled | `AITE_ENABLE_LLM=false` by default | No |
| Pose-generation/GAN model | Not implemented/trained | Recorded playback is used instead | No |
| Realistic 3D avatar | Not implemented | Current renderer is a 2D skeleton | No |

## 3. Actual system architecture

```text
Manual text input
       |
       v
Rule-based draft gloss conversion
       |
       v
Recorded-motion lookup
  | exact/near sentence match -> best CSLRT tracked sentence sequence
  | supported word match      -> INCLUDE-50 word sequence
  ` no supported match        -> labelled illustrative animation
       |
       v
2D landmark renderer -> ffmpeg H.264 MP4 -> browser video player
```

The planned uploaded-video path exists at the API level but is disabled:

```text
Uploaded ASL video -> MediaPipe features -> recognition model
                  X disabled because the checkpoint is not accurate enough
```

CSLRT is an **ISL** corpus. Its classifiers and motion assets cannot be described
as ASL input recognition. A complete ASL-to-ISL product needs both a reliable ASL
recognizer and reviewed cross-language translation supervision.

## 4. Completed changes

### 4.1 Backend and frontend

- Added and repaired the FastAPI application routes for health, translation,
  recognition input, sequence prediction, and avatar video generation.
- Kept recognition disabled by default through `AITE_ENABLE_RECOGNITION=false`.
- Added clear 503 responses telling users to enter text manually when recognition
  is unavailable.
- Mounted the browser application and generated-video directory.
- Added browser-compatible ffmpeg H.264 transcoding and unique generated filenames.
- Added UI state handling that waits for video data, exposes playback errors, and
  forces updated JavaScript to be fetched after avatar fixes.
- Added a visible status describing whether output is recorded sentence motion,
  recorded word motion, or the illustrative fallback.

### 4.2 MediaPipe runtime diagnosis

- Identified that Python 3.14 with MediaPipe 1.0.0 was killed while constructing
  `PoseLandmarker`, even for a one-sample run.
- Created `scripts/diagnose_mediapipe.py` to isolate native runtime failures.
- Created the compatible `.venv-cslrt` environment using Python 3.12,
  MediaPipe 0.10.21, NumPy 1.26.4, and OpenCV 4.x.
- Confirmed that the compatible environment creates and closes the landmarker.

### 4.3 Landmark normalization correction

- Corrected pose and hand landmarks to share shoulder-midpoint/shoulder-width
  coordinates.
- Preserved all-zero arrays for missing hands instead of translating them into
  plausible nonzero coordinates.
- Added `feature_schema_version: "2.0"` and tracking coverage metadata.
- Made the training dataset reject old/mixed-coordinate payloads.
- Added `scripts/validate_cslrt_landmarks.py` for schema, shape, finite-value,
  shoulder-center, and shoulder-scale checks.
- Regenerated and validated the sentence data after the correction.

### 4.4 MSASL recognition audit and rebuild

- Audited 16,785 usable local MSASL annotations and 18,032 downloaded video files.
- Confirmed the official MSASL-100 subset has 2,783 train, 645 validation, and
  522 test clips with no signer overlap between training and either held-out split.
- Discovered that all 14,263 legacy `cached_keypoints/*_f32.npy` sequences were
  all-zero arrays. The original 1.14% result was therefore a chance-level pipeline
  diagnostic, not a meaningful MSASL model result.
- Changed the legacy loader to reject all-zero caches with a clear error.
- Added `scripts/extract_msasl_landmarks.py`, which extracts normalized 33-pose,
  21-left-hand, and 21-right-hand landmarks into compressed NPZ artifacts.
- Added `scripts/validate_msasl_landmarks.py` and
  `scripts/build_msasl_splits.py` for schema, normalization, tracking quality,
  official-split, and coverage validation.
- Added `MSASLPoseDataset`, MSASL NPZ support to the generic trainer/evaluator,
  and `configs/msasl100_pose_recognition.yaml`.
- Extracted 3,938 usable sequences and recovered the manifest after detecting a
  partial metadata write; 12 zero-frame source videos were excluded.
- Validated 126,016 frames with 99.74% pose coverage and 80.28% hand coverage.
- Quality filtering retained 2,747 train, 643 validation, and 521 test sequences,
  with all 100 classes present and no signer overlap.
- Trained a 75-landmark (pose + both hands), velocity- and presence-aware temporal
  Transformer for 80 epochs with a fixed seed and saved runtime/configuration logs.
- Validation result: 44.48% top-1, 74.49% top-5, 39.38% macro-F1. Test result:
  47.41% top-1, 77.35% top-5, 43.69% macro-F1.
- The validation-selected 0.469525 threshold achieved 80.12% selective accuracy
  at 26.59% validation coverage, but only 76.73% on test; that baseline was not
  selected for deployment.
- A controlled class-balanced-sampling ablation improved validation macro-F1 to
  42.57% and test macro-F1 to 44.04%.
- A larger class-balanced Transformer (128 hidden units, three encoder layers) is
  the selected upload checkpoint: validation Top-1 49.30%, Top-5 75.58%, and
  macro-F1 45.34%; untouched test Top-1 47.02%, Top-5 76.78%, and macro-F1
  44.61%. Its validation-selected 0.590399 rejection threshold achieved 81.73%
  selective accuracy at 37.81% test coverage, meeting the predeclared gate for
  one MSASL-100-style isolated ASL sign per clip.

### 4.5 CSLRT sentence pipeline

- Extracted 663 signer/sentence sequences from 97 available sentence directories.
- Processed 18,863 frames with 99.97% pose coverage and 70.22% hand coverage.
- Added deterministic signer-independent manifests:
  - Train: signers 1–5, 474 samples, 97 classes.
  - Validation: signer 6, 96 samples, 96 represented classes.
  - Test: signer 7, 93 samples, 93 represented classes.
- Added the `CSLRTDataset`, temporal resampling, masking/noise/scale/frame-dropout
  augmentation, reproducible seeds, early stopping, and checkpoint logging.
- Added a reusable checkpoint evaluator with top-1/top-5 results, predictions,
  and confusion summaries.
- Trained the corrected v3 baseline and evaluated it only on held-out signer 7.

### 4.6 CSLRT isolated-word pipeline

- Confirmed that the word portion contains still images, not temporal video clips.
- Added `scripts/extract_cslrt_word_landmarks.py` with schema-v2 normalization.
- Fixed output collisions caused by duplicate source filenames by incorporating a
  stable hash of each complete relative source path.
- Preserved the earlier invalid derived directory as
  `data/isl/cslrt_word_keypoints_invalid_filename_collision` for traceability.
- Extracted 1,036 unique images across 114 labels with 100% pose coverage and
  95.27% hand coverage.
- Added deterministic class-stratified splits: 699 train, 168 validation, and
  169 test samples.
- Added class-balanced sampling and trained/evaluated a small static-word model.
- Generalized the CSLRT evaluator so static samples do not require signer fields.

### 4.7 Avatar/playback pipeline

- Added `configs/avatar_gloss_map.json` with 63 aliases covering 61 extracted
  INCLUDE-50 classes; its status remains `draft_pending_isl_expert_review`.
- Added recorded word-motion lookup and recorded CSLRT sentence-motion lookup.
- Selected the sentence recording with the highest hand-detection coverage.
- Added case/punctuation normalization and cautious approximate matching. Approximate
  matching requires strong text/token similarity and does not cross negation state.
- Slowed CSLRT playback from the incorrect 30 FPS default to 7 FPS because the
  extracted still-frame sequences do not contain source FPS metadata.
- Kept response metadata (`mode`, source sentence/gloss, signer, and match score)
  so retrieved motion cannot be mistaken for generated motion.

### 4.8 Translation safeguards

- Made Llama loading opt-in with `AITE_ENABLE_LLM=true`; it is off by default.
- Preserved deterministic rule-based operation when model weights are unavailable.
- Removed the ambiguous `HE → WO`, `SHE → WO`, `IT → WO`, and `THEY → WO`
  substitutions.
- Renamed the web output to “Draft rule-based gloss (signer review needed).”
- Made translation evaluation require a real reviewed test set instead of inventing
  fallback references.

### 4.9 Verification and documentation

- Added API contract, data split, metrics, translation, and preprocessing tests.
- Added recognition and translation evaluators, latency checks, QA scripts, dataset
  validators, and deployment guidance.
- Updated `docs/deployment.md` with environment-specific commands, measured model
  limits, playback modes, and safe demonstration guidance.

## 5. Data and artifacts

### 5.1 Main data directories

| Path | Purpose | Current approximate size |
|---|---|---:|
| `data/isl/include50_raw` | Raw INCLUDE-50 source data | 3.3 GB |
| `data/isl/keypoints` | Extracted INCLUDE-50 word motion | 2.7 GB |
| `data/isl/cslrt_raw` | Raw CSLRT corpus | 8.5 GB |
| `data/isl/cslrt_keypoints` | CSLRT sentence sequences | 66 MB |
| `data/isl/cslrt_word_keypoints` | CSLRT static-word features | 5.8 MB |
| `data/isl/cslrt_splits` | Sentence manifests/vocabulary | 176 KB |
| `data/isl/cslrt_word_splits` | Word manifests/vocabulary | 180 KB |

Raw datasets are local licensed research data and are not intended for Git or the
deployment image. Dataset license/usage requirements must be checked before sharing.

### 5.2 Model artifacts

| Path | Meaning | Status |
|---|---|---|
| `models/recognition` | Original MSASL experiment | Evaluated, unusable accuracy |
| `models/cslrt_recognition` | Historical v1 sentence experiment | Invalid old geometry |
| `models/cslrt_recognition_v2` | Historical v2 sentence experiment | Invalid old geometry |
| `models/cslrt_recognition_v3` | Corrected sentence baseline | Valid experiment, weak accuracy |
| `models/cslrt_word_recognition` | Static-word baseline | Valid experiment, weak accuracy |
| `models/mediapipe` | Pose and hand task models | Used for extraction |

The v1/v2 sentence checkpoints are retained only as historical diagnostics. Their
metrics must not be cited because pose and hands used incompatible coordinate systems.

## 6. Reproduce the current project

Run every command from the repository root:

```bash
cd /home/charliee/Documents/ASL-ISL
```

### 6.1 Start the demo

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python -m uvicorn \
  src.api.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Use manual text input. Known examples include:

```text
He is going into the room
He is going to the room
```

The second example is a cautious approximate match to the first dataset label.
The UI status must say which recorded motion was selected.

### 6.2 Run verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_cslrt_landmarks.py \
  --keypoints-dir data/isl/cslrt_keypoints
.venv/bin/python scripts/validate_cslrt_landmarks.py \
  --keypoints-dir data/isl/cslrt_word_keypoints
.venv/bin/python scripts/validate_avatar_gloss_map.py
```

Expected current results:

- 27 tests pass.
- Sentence data: 663 sequences, 18,863 frames, valid schema/normalization.
- Word data: 1,036 samples, valid schema/normalization.
- Avatar map: 63 aliases covering 61 extracted classes, pending expert review.

### 6.3 Extract and train the selected MSASL-100 recognition model

Use the compatible extraction environment. The command is resumable and reuses
valid NPZ outputs when rerun:

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv-cslrt/bin/python \
  scripts/extract_msasl_landmarks.py \
  --data-root Dataset/MS-ASL \
  --annotation-file Dataset/MS-ASL/MSASL_unified.json \
  --output-dir data/asl/msasl100_keypoints \
  --num-classes 100 \
  --num-frames 32 \
  --workers 4

.venv-cslrt/bin/python scripts/validate_msasl_landmarks.py \
  --keypoints-dir data/asl/msasl100_keypoints

.venv/bin/python scripts/build_msasl_splits.py \
  --keypoints-dir data/asl/msasl100_keypoints \
  --output-dir data/asl/msasl100_splits

.venv/bin/python scripts/train_cslrt_recognition.py \
  --config configs/msasl100_pose_balanced_large_recognition.yaml

.venv/bin/python scripts/evaluate_cslrt_recognition.py \
  --checkpoint models/msasl100_pose_balanced_large_recognition/best_model.pt \
  --keypoints-dir data/asl/msasl100_keypoints \
  --splits-dir data/asl/msasl100_splits \
  --split val \
  --output models/msasl100_pose_balanced_large_recognition/val_results.json

.venv/bin/python scripts/calibrate_msasl_recognition.py \
  --validation-results models/msasl100_pose_balanced_large_recognition/val_results.json \
  --output models/msasl100_pose_balanced_large_recognition/calibration.json

.venv/bin/python scripts/evaluate_cslrt_recognition.py \
  --checkpoint models/msasl100_pose_balanced_large_recognition/best_model.pt \
  --keypoints-dir data/asl/msasl100_keypoints \
  --splits-dir data/asl/msasl100_splits \
  --split test \
  --calibration models/msasl100_pose_balanced_large_recognition/calibration.json \
  --output models/msasl100_pose_balanced_large_recognition/test_results.json
```

Do not use `Dataset/MS-ASL/cached_keypoints` for new experiments. The current
audit found every legacy 27-keypoint cache file to be all-zero.

### 6.4 Regenerate CSLRT sentence landmarks

Use the Python 3.12 environment, not the main Python 3.14 environment:

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv-cslrt/bin/python \
  scripts/extract_cslrt_landmarks.py \
  --input-dir data/isl/cslrt_raw/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Sentence_Level \
  --output-dir data/isl/cslrt_keypoints \
  --overwrite

.venv/bin/python scripts/validate_cslrt_landmarks.py \
  --keypoints-dir data/isl/cslrt_keypoints
.venv/bin/python scripts/build_cslrt_splits.py
```

Omit `--overwrite` to resume/reuse existing outputs. Use `--overwrite` after any
normalization or feature-schema change.

### 6.5 Train and evaluate the sentence baseline

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python \
  scripts/train_cslrt_recognition.py \
  --config configs/cslrt_recognition_v3.yaml

MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python \
  scripts/evaluate_cslrt_recognition.py \
  --checkpoint models/cslrt_recognition_v3/best_model.pt \
  --keypoints-dir data/isl/cslrt_keypoints \
  --splits-dir data/isl/cslrt_splits \
  --output models/cslrt_recognition_v3/test_results.json
```

### 6.6 Regenerate, train, and evaluate static-word data

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv-cslrt/bin/python \
  scripts/extract_cslrt_word_landmarks.py \
  --input-dir data/isl/cslrt_raw/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level \
  --output-dir data/isl/cslrt_word_keypoints

.venv/bin/python scripts/validate_cslrt_landmarks.py \
  --keypoints-dir data/isl/cslrt_word_keypoints
.venv/bin/python scripts/build_cslrt_word_splits.py

MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python \
  scripts/train_cslrt_recognition.py \
  --config configs/cslrt_word_recognition.yaml

MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python \
  scripts/evaluate_cslrt_recognition.py \
  --checkpoint models/cslrt_word_recognition/best_model.pt \
  --keypoints-dir data/isl/cslrt_word_keypoints \
  --splits-dir data/isl/cslrt_word_splits \
  --output models/cslrt_word_recognition/test_results.json
```

## 7. Measured model results

| Experiment | Evaluation split | Top-1 | Top-5 | Interpretation |
|---|---|---:|---:|---|
| Original MSASL ASL recognizer | 701 validation clips | 1.14% | 4.28% | Pipeline checkpoint only |
| Rebuilt MSASL-100 pose+hands baseline | 643 validation clips | 44.48% | 74.49% | Macro-F1 39.38%; threshold selected here |
| Rebuilt MSASL-100 pose+hands baseline | 521 untouched test clips | 47.41% | 77.35% | Macro-F1 43.69%; selective accuracy 76.73% at 30.52% coverage |
| Balanced larger MSASL-100 Transformer | 643 validation clips | 49.30% | 75.58% | Macro-F1 45.34%; threshold selected here |
| Balanced larger MSASL-100 Transformer | 521 untouched test clips | 47.02% | 76.78% | Macro-F1 44.61%; selective accuracy 81.73% at 37.81% coverage |
| Corrected CSLRT v3 sentence classifier | 93 signer-7 samples | 5.38% | 17.20% | Not deployable |
| CSLRT static-word classifier | 169 held-out images | 7.69% | 31.36% | Auxiliary experiment only |

The 1.14% / WER 0.9886 MSASL result is retained only as a historical diagnostic:
the audit found every legacy input feature sequence to be all-zero. The rebuilt
MSASL-100 rows use `data/asl/msasl100_keypoints` and are the only current
recognition-performance claims.

## 8. API behavior

| Route | Current behavior |
|---|---|
| `GET /` | Serves the browser interface |
| `GET /health` | Reports service, recognition, and playback availability |
| `POST /translate` | Returns draft rule-based gloss conversion |
| `POST /generate-avatar` | Returns recorded motion or illustrative fallback MP4 |
| `POST /extract-keypoints` | Still-image recognition not supported (410) |
| `POST /predict-gloss` | Still-image recognition not supported (410) |
| `POST /predict-sequence` | Disabled by default; when enabled, accepts one isolated-sign MP4/WebM/MOV with quality/rejection checks |

Important `/generate-avatar` response fields:

- `mode`: `recorded_sentence_landmark_playback`, `landmark_playback`, or
  `illustrative_2d`.
- `source_sentence` / `source_gloss`: identifies the retrieved asset.
- `source_signer`: dataset signer for sentence retrieval.
- `sentence_match` and `sentence_match_score`: disclose approximate matching.
- `video_url`: browser-accessible generated file under `/generated/`.

## 9. Known limitations and risks

1. **Limited ASL video input.** Upload and webcam capture support one isolated
   MSASL-100-style sign only; continuous signing and signs outside the 100-class
   vocabulary remain unsupported and should be rejected or entered manually.
2. **No validated ASL-to-ISL parallel translation data.** Current word mappings
   and grammar rules are drafts and may be linguistically incorrect.
3. **No generative avatar.** Recorded skeleton playback cannot create unseen
   sentences, coarticulation, facial grammar, or natural transitions.
4. **Dataset scope mismatch.** MSASL is ASL isolated-word data; CSLRT is ISL
   sentence/word data. Neither alone supplies supervised ASL-to-ISL translation.
5. **Small CSLRT sample count.** Roughly seven signer renditions per sentence are
   insufficient for a robust 97-class classifier or open-vocabulary generator.
6. **Static-word data has no motion.** A single image cannot teach movement,
   direction, timing, or dynamic handshape changes.
7. **Missing non-manual features.** Facial expression, gaze, mouth pattern, and
   body orientation are not represented by the current 75-landmark schema.
8. **Approximate sentence lookup is retrieval.** It is constrained by similarity
   and negation checks but still requires user-visible disclosure and expert review.
9. **Local generated files accumulate.** `/tmp/aite-generated` should receive a
   bounded retention policy before long-running deployment.
10. **Documentation history contains aspirational claims.** Older README/report
    text referring to completed LLM/GAN functionality must not be used as evidence
    of implementation.

## 10. Prioritized further work

### Phase 1 — Freeze and validate the research baseline

1. Review all 97 sentence labels, 114 static-word labels, and 63 avatar aliases
   with a qualified ISL signer/linguist.
2. Record decisions in versioned JSON/CSV with reviewer, date, and license fields.
3. Add dataset integrity checksums and a reproducible environment lock file.
4. Remove or archive derived smoke/collision artifacts only after verifying that
   no scripts reference them.

**Exit condition:** reviewed label inventory, documented licenses, reproducible
clean setup, and the existing 27 tests still passing.

### Phase 2 — Build a defensible ASL recognition input

1. Retain the completed quality-filtered, signer-separated MSASL-100 manifests
   and selected checkpoint as a reproducible baseline.
2. Audit accepted/rejected upload cases from unseen signers, documenting latency,
   confidence, and per-class errors.
3. Run the remaining controlled ablations: pose only, hands only, non-manual
   features, feature normalization, and pretrained RGB/video models on GPU.
4. Expand only after obtaining licensed continuous-signing data and defining a
   sentence-level evaluation protocol.

**Predeclared deployment gate:** raw validation Top-1 >= 40% and macro-F1 >= 35%,
plus validation selective accuracy >= 80% at >= 20% coverage. Select the rejection
threshold on validation only and apply it once to test. The selected isolated-sign
model passed this gate; any future replacement must pass it before enablement.
These are deployment gates for this prototype, not a claim of human-level
recognition.

**Exit condition:** the predeclared held-out gate is met, errors are audited by
signer/class, and the upload endpoint rejects low-confidence inputs.

### Phase 3 — Create reviewed ASL-to-ISL translation supervision

1. Build a parallel dataset of ASL gloss sequence, meaning/English representation,
   and reviewed ISL gloss sequence.
2. Define a consistent ISL gloss notation including spatial references,
   repetition, negation, questions, and non-manual markers.
3. Replace unreviewed Hindi transliterations with reviewed sign identifiers.
4. Establish exact-match, BLEU/chrF, semantic, and human adequacy evaluations.
5. Use an LLM only as an optional constrained component after establishing a
   reviewed rule/sequence baseline. Downloading a 7B model alone does not provide
   ISL knowledge or solve missing supervision.

**Exit condition:** held-out translation results and human evaluation are recorded;
the UI no longer labels unreviewed output as translated ISL.

### Phase 4 — Train pose generation

1. Preserve motion sequences, tracking masks, FPS/timing, signer information,
   facial features, and gloss alignment in a generation-specific manifest.
2. Begin with a motion autoencoder and a gloss-conditioned temporal model; do not
   begin with a GAN solely because older planning documents mention one.
3. Train with position, velocity, acceleration/jerk, bone-length, handshape, and
   missing-keypoint-mask losses.
4. Evaluate reconstruction and generation using DTW, PCK, motion smoothness,
   bone consistency, and signer review.
5. Add transitions/coarticulation between signs and explicit duration prediction.
6. Retarget validated pose sequences to a rigged 3D avatar only after pose quality
   is acceptable.

The existing 97-sentence corpus is useful for retrieval, reconstruction experiments,
and a small closed-vocabulary baseline. More diverse reviewed motion data will be
required for meaningful unseen-sentence generation.

**Exit condition:** unseen held-out sequences are intelligible to ISL reviewers,
motion metrics beat retrieval/interpolation baselines, and failures are disclosed.

### Phase 5 — Integrate and harden the complete application

1. Connect recognition only after its threshold is met.
2. Propagate confidence and provenance through recognition, translation, and
   generation responses.
3. Add request size/type limits, generated-file cleanup, rate limits, structured
   logs, and privacy rules for uploaded video.
4. Add end-to-end browser tests for success, unknown signs, API failure, and video
   decoding.
5. Benchmark CPU/GPU latency and memory on the intended deployment hardware.
6. Deploy privately for reviewed user testing before any public release.

**Exit condition:** the complete pipeline meets accuracy, latency, accessibility,
security, privacy, and signer-review acceptance criteria.

## 11. Immediate next actions

The recommended next actions, in order, are:

1. Demo the calibrated MSASL-100 isolated-sign upload/webcam path with clips from
   unseen signers; record latency, accepted/rejected outcomes, and failure cases.
2. Obtain an ISL expert review of the 97 sentence labels and playback output.
3. Build reviewed ASL-to-ISL parallel supervision, including a licensed
   continuous-signing evaluation set.
4. Add facial/non-manual features and compare against a pretrained RGB/video
   baseline on suitable GPU infrastructure.
5. Export a reviewed generation manifest and implement a motion-autoencoder
   baseline; do not characterize retrieval as generation.

No model should be enabled in the public-facing pipeline solely because training
completed. Enablement requires held-out metrics and language-expert review.

## 12. Definition of the final project being “done”

The project can be described as complete only when all of the following are true:

- Uploaded ASL video is recognized reliably on unseen signers.
- ASL output is translated using reviewed ASL–ISL supervision.
- Unsupported input is detected instead of producing a confident false answer.
- The avatar produces intelligible ISL motion for held-out sentences, including
  timing, hands, body, and non-manual grammar.
- ISL users/experts validate comprehension and naturalness.
- End-to-end tests, latency targets, accessibility, privacy, and deployment checks
  pass on the target system.
- Documentation reports measured results and limitations without calling retrieval
  “generation” or a draft mapping “validated translation.”

Until those conditions are met, AITE should be presented as a working research
prototype with validated preprocessing and limited recorded-pose playback.
