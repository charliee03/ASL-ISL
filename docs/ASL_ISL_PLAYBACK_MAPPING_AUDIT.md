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

## Experimental recognition evidence; no additional playback paths

An offline 126-class experimental recognizer added 26 exact-English-label
INCLUDE-50 candidates to the 100-class MSASL vocabulary. It was evaluated with
signer-disjoint data, but is deliberately not connected to the API or renderer.
The candidate classes are `bear`, `busy`, `carrot`, `close`, `cook`, `cry`,
`deer`, `exam`, `giraffe`, `give`, `good morning`, `jump`, `key`, `knife`,
`lion`, `maybe`, `monkey`, `onion`, `still`, `tea`, `tiger`, `turtle`,
`umbrella`, `uncle`, `wife`, and `wrong`. The requested `break` and `elephant`
classes could not be retained because they were absent from at least one
quality-filtered held-out split.

Per-class held-out recognition evidence is in
`docs/msasl126_candidate_class_metrics.json`. Several classes have one to six
test clips, so those values are triage evidence, not an adequacy claim. Every
candidate remains blocked from playback until an ISL signer/linguist confirms
the semantic equivalence, the specific INCLUDE-50 recording, and the intended
use case. An English-string match is not such confirmation.

The ready-to-use first-batch review sheet is
[`ASL_ISL_MAPPING_REVIEW_TEMPLATE.csv`](ASL_ISL_MAPPING_REVIEW_TEMPLATE.csv),
with instructions in
[`ASL_ISL_MAPPING_REVIEW_GUIDE.md`](ASL_ISL_MAPPING_REVIEW_GUIDE.md).

## Unconnected INCLUDE-50 targets

The following targets have no exact, evaluated candidate source label. They
require both (1) an approved ASL-to-ISL semantic mapping and (2) licensed ASL
examples in the recognition vocabulary before they can be enabled:

```text
brinjal, budget, cabbage, cauliflower, chilli, clean, come, crocodile, cucumber,
fedup, fever, good_afternoon, hug, injury, interview, karnataka, lemon, maths,
peacock, pigeon, pour, radish, sparrow, switch, temple, thank_you, vegetables,
volcano, what_is_your_name, writer
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
