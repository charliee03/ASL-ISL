# AITE: IEEE paper-writing handoff

Status: 11 September 2026. Audience: the teammate writing the paper and the project team. This is the current documentation entry point; older reports describe earlier stages, not necessarily current capabilities.

## 1. Read these first

1. This guide: scope, methods, evidence, limitations, and writing plan.
2. [Paper draft](main.tex): editable IEEEtran source; a starting draft, not a submission-ready manuscript. The existing `main.pdf` predates this handoff and must not be treated as the updated paper.
3. [Evidence export](paper_evidence/README.md): machine-readable metrics, file hashes, and reproducible snapshot command.
4. [Translation implementation](TRANSLATION.md), [deployment commands](deployment.md), and [demo acceptance plan](DEMO_RUNBOOK.md).
5. [Mapping review guide](ASL_ISL_MAPPING_REVIEW_GUIDE.md), [review sheet](ASL_ISL_MAPPING_REVIEW_TEMPLATE.csv), [mapping audit](ASL_ISL_PLAYBACK_MAPPING_AUDIT.md), and [candidate class analysis](msasl126_candidate_class_metrics.json).
6. [Dataset acquisition decision](DATASET_ACQUISITION_DECISION.md) and [historical experiment/roadmap record](PROJECT_STATUS_AND_ROADMAP.md).

No datasets, checkpoints, teammate changes, or historical reports were deleted for this handoff. Large local artifacts are not necessarily tracked in Git. A repository clone alone is not a complete reproducibility archive.

## 2. Defensible scope and contribution

Suggested title: **AITE: A Reproducible Prototype for Isolated ASL Recognition and Recorded ISL Pose Playback**.

AITE is a modular local research prototype combining isolated ASL recognition, a draft text/gloss conversion layer, confidence rejection, and retrieval of recorded ISL two-dimensional landmark motion. It also contains an offline motion-reconstruction experiment. The contribution is the implemented integration and documented evaluation, not an established new translation algorithm or state-of-the-art recognition result.

The following distinctions must remain visible throughout the paper:

| Component | Evidence available | Do not claim |
| --- | --- | --- |
| MSASL-100 recognition | Saved held-out classification and selective-prediction results | Continuous signing, general webcam accuracy, or 81.73% overall accuracy |
| MSASL-126 candidate | Separate saved recognition experiment | Proven improvement on the same benchmark or 126 reviewed ISL outputs |
| Rule translation | Implemented mapping/filtering and automated behavior tests | Linguistically validated ASL-to-ISL sentence translation |
| Gemini/Llama | Optional draft refinement code, disabled by default | Measured accuracy gain, locally trained translation model, or semantic guarantees |
| Recorded word/sentence playback | Local pose assets and historical engineering demo report | Generated signs, a realistic 3D avatar, or complete multiword output |
| Fingerspelling | Teammate code and mock-based tests | Verified NANDITA spelling without real alphabet assets and visual review |
| Motion autoencoder | Saved reconstruction losses | Text-to-sign generation or signer comprehension |
| Browser/webcam | UI implementation | Physical-camera acceptance completed |

AI review is a preliminary engineering/label triage, not an ISL signer endorsement. The mapping sheet's `ai_review_pending_expert` and `not_signer_reviewed` states must remain explicit. Academic/internship scope does not turn this into expert validation.

## 3. Architecture and implementation map

```text
Manual text ----------------------------------------+
                                                    v
Video / webcam -> upload -> landmark extraction -> isolated classifier
                                      |             |
                                      |       reject low confidence
                                      |             |
                                      +---------- accepted ASL gloss
                                                    |
                                      deterministic draft conversion
                                                    |
                                   optional Gemini / Llama refinement
                                                    |
                                    recorded-motion lookup / fallback
                                                    |
                                      2D MP4 + provenance in browser

CSLRT poses -> offline motion autoencoder -> reconstruction metrics
              (not connected to the runtime renderer)
```

| Area | Source of truth |
| --- | --- |
| API routes, flags, upload checks, selection and retention | `src/api/server.py` |
| Browser recording, API calls, labels | `src/web/app.js`, `index.html`, `styles.css` |
| Recognition training/evaluation | `scripts/train_cslrt_recognition.py`, `evaluate_cslrt_recognition.py` |
| MSASL preparation | `scripts/build_msasl_splits.py`, `extract_msasl_landmarks.py`, `validate_msasl_landmarks.py` |
| Validation threshold | `scripts/calibrate_msasl_recognition.py` |
| Gloss conversion and model prompting | `src/translation/translator.py`, translation configuration under `configs/` |
| Motion reconstruction | `src/generation/motion_autoencoder.py`, `scripts/train_motion_autoencoder.py` |
| Playback/fingerspelling | `src/generation/`, `configs/avatar_gloss_map.json`, `data/isl/` |
| Tests | `tests/`; these do not replace live browser or linguistic tests |

Runtime endpoints include `/health`, `/translate`, `/predict-sequence`, `/generate-avatar`, and `/generated/…`. `/health` reports service/model state, not translation quality or successful provider access.

## 4. Data inventory and provenance

Counts below distinguish saved experiment sample counts from historical extraction inventory. Before submission, regenerate split/quality summaries and freeze the exact files used.

| Data | Role and local location | Recorded scope | Qualification |
| --- | --- | --- | --- |
| MSASL subset | ASL classification; `Dataset/MS-ASL/`, `data/asl/msasl100_keypoints`, `data/asl/msasl100_splits` | Selected model: 2,747 train / 643 val / 521 test, 100 classes | Filtered local subset, not the full published dataset |
| MSASL expansion | `data/asl/msasl126_splits` with same feature directory | 3,062 / 729 / 593, 126 classes | Adds 26 ASL labels selected by English overlap with ISL labels; does not train ASL recognition on ISL clips |
| INCLUDE-50 local collection | ISL recorded word playback; `data/isl/include-50`, `data/isl/keypoints` | Historical inventory: 61 word classes, 63 aliases | Local extraction count is not a claim about official release size or reviewed equivalence |
| ISL-CSLRT sentence data | `data/isl/cslrt_raw`, `data/isl/cslrt_keypoints`, `data/isl/cslrt_splits` | 663 sequences / 97 sentence labels; 474 / 96 / 93 split | Recorded signer split: 1–5 train, 6 val, 7 test; sentences are not aligned ASL translations |
| ISL-CSLRT word images | Static-word baseline | Historical 1,036 images / 114 labels; 699 / 168 / 169 | Static images cannot establish dynamic-sign performance |
| Alphabet keypoints | `data/isl/alpha_keypoints/isl_alpha_*.json` | Awaiting teammate asset delivery and acceptance | Mock tests are not an asset inventory |
| iSign | Metadata/group-split preparation only | Full acquisition deferred | Not used to produce the reported main recognition or translation results |

There is no verified parallel ASL–ISL sentence training/test corpus in this evidence package. The earlier approximately 228 GB discussion concerns a dataset acquisition decision, not a required 200 GB LLM. Review the acquisition document before allocating storage; its size estimate is historical, not a fresh download-size measurement.

Dataset rights and provenance are a submission gate. Record the exact release URL/version, access date, license/permission, local filters, signer split, exclusions, and redistribution conditions for each collection. In particular, the precise local INCLUDE/CSLRT source and permissions need team confirmation. Do not describe possession of files as proof of a license. Do not publish raw clips, face images, or derivative landmarks without checking applicable permissions and consent. Keep a private source-to-feature manifest for reproducibility.

## 5. Recognition method and training protocol

The selected configuration is `configs/msasl100_pose_balanced_large_recognition.yaml`, with the resolved configuration and runtime metadata beside the checkpoint.

- Input: 32 temporal samples; 33 body + 21 left-hand + 21 right-hand points = 75 points with three coordinates. Shoulder midpoint/width normalization; missing detections require explicit quality checks.
- Features: flattened position (225), velocity (225), and presence (75): 525 input channels when both options are enabled.
- Encoder: projection to width 128, sinusoidal temporal position encoding, three Transformer layers, four attention heads, feed-forward width 512, dropout 0.2, learned-query attention pooling, 100-class head. Saved trainable parameter count: 741,476 (126-class variant: 744,830).
- Training configuration: seed 42, batch 64, class-balanced sampling, maximum 80 epochs, early-stopping patience 15, AdamW with learning rate 0.0003 and weight decay 0.0005, label smoothing 0.05, cosine scheduling. Maximum epochs is a configuration budget, not necessarily the number actually completed; inspect `training_log.csv`.
- Augmentations: minimum temporal-crop ratio 0.9, scale amount 0.03, noise standard deviation 0.003, keypoint masking 0.01 and frame dropping 0.01. Do not apply training augmentation to validation/test.
- Checkpoint selection uses validation accuracy. Confidence threshold selection uses validation predictions with target selective accuracy 0.80 and minimum coverage 0.20. Test results are then computed using that fixed threshold.
- Saved runtime reports CPU training, Python 3.14.7 and PyTorch 2.13.0+cu130. A CUDA-tagged build does not mean training used a GPU. Record hardware and wall time separately; they are not established here.

The encoder is SPOTER-inspired; do not label it an exact SPOTER reproduction. Official MSASL partitions motivate signer-independent evaluation, but local split identities, filtering, and duplicate/source leakage should be audited before the final manuscript. The recorded protocol is validation-based selection; saved files alone cannot prove the test set was never consulted during all development.

## 6. Results suitable for tables

Source: saved `val_results.json`, `test_results.json`, and `calibration.json` in the respective model directories. These are archived experiment results, not newly retrained measurements. Metrics below are percentages except threshold, ECE and NLL.

| Model / split | N | Top-1 | Top-5 | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| MSASL-100 validation | 643 | 49.30 | 75.58 | 45.34 |
| MSASL-100 test | 521 | 47.02 | 76.78 | 44.61 |
| MSASL-126 validation | 729 | 45.13 | 72.43 | 42.73 |
| MSASL-126 test | 593 | 50.42 | 76.39 | 45.99 |

| Model, test split | Validation threshold | Accepted / total | Coverage | Accuracy among accepted |
| --- | ---: | ---: | ---: | ---: |
| MSASL-100 | 0.5903989673 | 197 / 521 | 37.81 | 81.73 |
| MSASL-126 | 0.6365509629 | 168 / 593 | 28.33 | 84.52 |

Selected MSASL-100 test macro precision: 47.51%; macro recall/balanced accuracy: 48.47%; ECE: 0.09803; NLL: 2.20967. The 126-class test ECE is 0.05030 and NLL is 2.21032. Read the evaluator for the precise ECE binning and averaging conventions before defining them in the manuscript.

Coverage = accepted predictions / all examples. Selective accuracy = correct accepted predictions / accepted predictions. The 81.73% figure excludes 324 rejected clips; it is not overall translation accuracy or an open-set recognition guarantee. A softmax threshold can still accept an unknown sign confidently.

The 100- and 126-class tests have different labels and sample populations. Their Top-1 values are not a controlled improvement comparison. No multi-seed uncertainty interval, statistical significance result, matched-input Gemini ablation, or translation BLEU/semantic-adequacy result is established.

### Secondary experiments (keep separate from the main result)

| Experiment | Test samples | Top-1 | Top-5 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| CSLRT sentence-label classifier v3 | 93 | 5.38% | 17.20% | Weak isolated sequence-label baseline, not sentence translation |
| CSLRT static word classifier | 169 | 7.69% | 31.36% | Weak static-image baseline, not a deployment result |

The offline motion autoencoder uses 48-frame sequences, hidden width 192, latent width 96, batch 16, seed 42, learning rate 0.001 and a 50-epoch configured budget. Objective: position MSE + 0.2 × velocity MSE. Saved validation/test losses are 0.19724 / 0.44795; position MSE 0.18640 / 0.42194; velocity MSE 0.05421 / 0.13006; MAE 0.25567 / 0.40341. These are normalized-coordinate reconstruction measures, not metres or sign accuracy. Missing-point zeros are included in the unmasked objective. The model reconstructs supplied poses; it is not gloss-conditioned and is not used by the API.

## 7. Translation, playback and unresolved behavior

See [TRANSLATION.md](TRANSLATION.md) for provider configuration and actual processing. These issues must be disclosed or fixed and retested before claiming broader capability:

1. **Meaning loss in rules:** filler filtering is context-free and can remove meaningful words such as LIKE or RIGHT. Configured grammar descriptions are not evidence of executed tense, classifier, or negation rules.
2. **Provider validation:** Gemini's allowlist contains mapping/alias tokens, not only signer-approved assets. Its additional `unsupported_source_tokens` field is appended after the main token check without equivalent source/completeness validation. No meaning-preservation or accuracy gain is proven.
3. **Incomplete multiword playback:** absent a sentence match, the normal recorded-word path can select the first recognized word and render only that word while displaying the whole input. A returned MP4 is not proof the whole sentence was signed.
4. **Approximate sentence lookup:** token overlap and sequence similarity, with a limited negation guard, can retrieve a different meaning. It is retrieval, not generative translation.
5. **Alphabet gaps:** missing letters can be skipped; a nonempty fingerspelling result does not establish complete spelling. Confirm all N-A-N-D-I-T-A frames, order, transitions, handedness and legibility after real assets arrive.
6. **Extraction runtime mismatch:** the current upload handler launches extraction using the API's Python executable. The historical verification helper uses `.venv-cslrt`. The older MediaPipe environment workaround therefore does not establish the current HTTP upload path is healthy.
7. **Browser MIME compatibility:** the server uses exact accepted content types; a browser value such as `video/webm;codecs=vp9` can be rejected. Camera stop/clear interactions also need physical-browser testing.
8. **Retention/security:** generated-video cleanup is request-triggered, not a timed deletion guarantee. Wildcard CORS and absence of authentication make this a local demo, not a hardened public service.

These are code-review findings, not all newly reproduced live failures. This documentation task does not silently change runtime behavior. Capture a failing case, fix in a separate implementation step, and update this list only with test evidence.

## 8. Reproduction and evidence capture

Run from the repository root. Keep the current trained outputs intact. Evaluation below writes to `/tmp`, not over archived results.

```bash
MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/pytest -q
.venv/bin/python scripts/export_paper_evidence.py
MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/python scripts/evaluate_cslrt_recognition.py \
  --checkpoint models/msasl100_pose_balanced_large_recognition/best_model.pt \
  --keypoints-dir data/asl/msasl100_keypoints \
  --splits-dir data/asl/msasl100_splits --split test \
  --calibration models/msasl100_pose_balanced_large_recognition/calibration.json \
  --output /tmp/aite-msasl100-test-recheck.json
```

Training and extraction commands are in [deployment.md](deployment.md). Retraining is optional for paper preparation and may be expensive. Copy configurations to a separate experiment output directory before running training; do not overwrite the selected checkpoint. Preserve validation predictions, calibration, test predictions, split files and RNG/runtime metadata together.

For a local manual-text smoke test, start with both provider flags off:

```bash
AITE_ENABLE_GEMINI=false AITE_ENABLE_LLM=false AITE_ENABLE_RECOGNITION=false \
  MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` in the browser on that same machine. A successful `/health` is only a service check. Use the acceptance log for actual rendering, upload and webcam observations. Recognition requires its flag and a validated compatible extraction environment; do not infer that from a passing Python test suite.

For optional Gemini, load the secret file only in the server terminal, explicitly enable the flag, and record model identifier, date, prompt/code hash and returned translation mode. Never paste a key into the paper, commands saved in Git, screenshots, browser code or this evidence archive. Rotate the key previously disclosed in chat. No live paid-provider call is required by this handoff.

Tests are functional/regression evidence, often using mocks/direct handlers. They do not measure linguistic quality, a real webcam, provider availability, complete browser playback, or deployment security. `demo_video_to_avatar_verification.json` is historical evidence for three selected helper-script examples, not a newly executed full-API benchmark or expert review. Use [DEMO_VALIDATION_LOG_TEMPLATE.csv](DEMO_VALIDATION_LOG_TEMPLATE.csv) for new observations; leave unexecuted rows blank.

## 9. Paper outline and figures

| Section | What the writer should include |
| --- | --- |
| Abstract | Narrow problem, implemented pipeline, test Top-1 and selective accuracy WITH coverage, principal limitation |
| Introduction | ASL/ISL distinction, isolated-sign scope, bounded integration contribution |
| Related work | Verified MSASL, SPOTER and MediaPipe sources; add verified ISL dataset papers after provenance confirmation |
| Method | Architecture, features, model, draft translation, retrieval and rejection logic; separate optional experiments |
| Data/protocol | Exact local subset, splits, quality filters, model selection, threshold selection, licenses |
| Results | Main and selective tables; secondary baselines separately; disclose different populations |
| Discussion | Failure cases, language-review gap, offline/provider tradeoffs, no causal Gemini gain claim |
| Ethics/limitations | Privacy, Deaf/signer involvement, consent, non-manual signals, local-only deployment |
| Conclusion | Measured bounded result and concrete remaining work |
| References | `paper_sources.bib` plus independently checked additions |

Use the architecture diagram above for Figure 1. Prepare a validation-only threshold/coverage plot from saved predictions if desired, and a clearly labelled real input/recorded-output figure with permissions. Do not fabricate screenshots or use `input_frame.jpg` / `output_skeleton.jpg` as measured evidence without identifying their provenance. Do not call an illustrative skeleton an ISL sign. The legacy `paper_sections/`, reports and `refs.bib` are background material requiring fact checking; some old citation metadata is unverified.

The small bibliography is checked against primary metadata: [MS-ASL](https://arxiv.org/abs/1812.01053), [SPOTER workshop paper](https://openaccess.thecvf.com/content/WACV2022W/HADCV/html/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.html), and [MediaPipe Hands](https://arxiv.org/abs/2006.10214). Gemini structured output constrains format, not semantic correctness; see [Google's structured-output documentation](https://ai.google.dev/gemini-api/docs/structured-output). Do not transfer published benchmark scores to this local subset.

## 10. Keep/share checklist

- Share source, tests, configs, this guide, the editable `.tex`, verified `.bib`, evidence export, review sheets and completed demo logs with the teammate.
- Privately back up selected and candidate model directories including checkpoints, vocabularies, resolved configs, runtime, training logs, validation/test predictions and calibration. Keep motion and weak-baseline results for honest reporting.
- Preserve exact split lists, feature manifests/extraction logs, dataset source/permission records and asset mappings. Hashes verify file identity, not scientific validity or historic execution.
- Do not put raw licensed data, keys, secret environment files, virtual environments, caches or user-identifying debug logs into a public paper repository. Sanitize evidence/screenshots before sharing. Do not delete local data simply because it is excluded from Git.
- Record code commit AND the uncommitted-change state: this handoff is based on a dirty working tree, so HEAD alone cannot reproduce it. Team-review and commit intended files separately; this documentation task does not commit or push.
- Confirm author order, affiliations, corresponding author, supervisor role, funding, acknowledgments and venue-specific template/page limits. Internship participation is not publication acceptance. Check the target venue's current AI-assistance/disclosure rules before submission and describe assistance accurately.

## 11. Remaining work / ownership proposal

| Priority | Task | Suggested owner | Completion evidence |
| --- | --- | --- | --- |
| Before broad demo claims | Fix/runtime-test extraction and MIME handling; test camera stop/clear | Integration owner + user | Browser/device/version and saved acceptance log |
| Before complete spelling claim | Deliver alphabet assets; test missing letters and NANDITA visually | Teammate + integration owner | Asset inventory and all-letter playback review |
| Before sentence-output claim | Resolve first-word-only playback and approximate-match ambiguity | Integration owner | Meaning-sensitive multiword/negation regression cases |
| Before translation-quality claim | Review mappings and outputs with competent ISL users | Team + ISL reviewer | Separate signed expert-review record, not AI approval |
| Before LLM improvement claim | Paired rule-vs-provider evaluation on held-out examples | Experiment owner | Fixed inputs, failure/latency/cost records and human adequacy rubric |
| Before submission | Verify licenses/citations/splits; freeze evidence; compile and proofread | Paper writer + team | Reviewed archive, compiled PDF and resolved checklist |

Webcam and teammate asset validation remain deferred as requested. Other open issues are documented above; the project should not currently be described as “everything else fully working.”
