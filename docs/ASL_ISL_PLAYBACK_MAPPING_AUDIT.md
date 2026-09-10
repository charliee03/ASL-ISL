# ASL-to-ISL playback mapping audit

## Purpose and evidence boundary

This audit records the connection between the current MSASL-100 ASL video
recognizer and the 61 extracted INCLUDE-50 ISL word-pose playback classes. It is
an engineering inventory, not linguistic validation. A row must not be enabled
for automatic video-to-avatar playback until a qualified ISL signer/linguist has
reviewed the proposed ASL-to-ISL equivalence.

The current recognizer is limited to 100 **isolated ASL** labels. INCLUDE-50 is
the target **ISL** motion source. It cannot be used as additional ASL recognition
training data.

## Connected paths

| MSASL-100 ASL result | Draft ISL gloss | INCLUDE-50 pose target | State |
|---|---|---|---|
| `hello` | `NAMASKAR` | `hello` | Implemented; linguistic review still required |
| `drink` | `DRINK` | `drink` | Implemented; linguistic review still required |
| `man` | `MAN` | `man` | Implemented; linguistic review still required |

These are the only current end-to-end paths that use a recorded ISL word-pose
asset after uploaded-video recognition. The recognizer can accept other ASL
classes, but the app must use a labelled illustrative fallback when no reviewed
recorded ISL motion target exists.

## Unconnected INCLUDE-50 targets

The following 58 targets have no exact, enabled MSASL-100 source-label mapping.
They require both (1) an approved ASL-to-ISL semantic mapping and (2) licensed
ASL examples in the recognition vocabulary before they can be enabled:

```text
bear, break, brinjal, budget, busy, cabbage, carrot, cauliflower, chilli, clean,
close, come, cook, crocodile, cry, cucumber, deer, elephant, exam, fedup, fever,
giraffe, give, good_afternoon, good_morning, hug, injury, interview, jump,
karnataka, key, knife, lemon, lion, maths, maybe, monkey, onion, peacock, pigeon,
pour, radish, sparrow, still, switch, tea, temple, thank_you, tiger, turtle,
umbrella, uncle, vegetables, volcano, what_is_your_name, wife, writer, wrong
```

`what_is_your_name` is a sentence-like concept, not an isolated-word class. It
must be handled as an approved sentence mapping and retrieved sentence motion,
not assembled by pretending that it is one isolated ASL classifier label.

## Required procedure per target

1. **Verify the ISL target.** An ISL signer/linguist confirms that the
   INCLUDE-50 label and recorded motion are appropriate for the intended meaning.
2. **Choose the ASL source concept.** Record the precise ASL gloss and any sense
   distinction; do not use English spelling as proof of sign equivalence.
3. **Secure ASL data.** Use licensed, multi-signer, isolated ASL clips for that
   gloss. If it is absent from MSASL-100, add data and retrain/evaluate the
   recognizer with signer-independent splits.
4. **Evaluate before mapping.** Measure per-class precision, recall, confidence
   calibration, and rejection behaviour on unseen signers.
5. **Enable only after review.** Add the approved translation rule and avatar
   alias, with dataset/source/reviewer/date evidence; add an integration test
   that verifies the route selects the intended pose target.

## Next implementation batch

Start with a small, high-value batch whose ASL sources and ISL meanings can be
reviewed quickly. Keep the initial batch to no more than five concepts. Candidate
selection must be driven by available licensed ASL clips and signer approval,
not by which English names happen to look similar.

Do not map `PLEASE -> KRIPAYA` to an unrelated INCLUDE-50 motion. Until a
licensed isolated ISL `PLEASE/KRIPAYA` motion recording is available, retain the
explicit illustrative fallback.
