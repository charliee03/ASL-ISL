# ASL–ISL mapping review package

This package prepares the next safe expansion of AITE's recorded-pose playback.
It does not approve any ASL–ISL mapping by itself. Only a qualified ISL
signer/linguist can do that.

## AI-assisted draft review

`ai_review_pending_expert` records a semantic triage prepared by an AI. It can
identify obvious English-label ambiguities and prioritise rows for review, but
it is not linguistic approval and cannot confirm that a recorded motion is an
appropriate ISL sign. Rows with this status remain disabled in the avatar API.

## What to review

Use [ASL_ISL_MAPPING_REVIEW_TEMPLATE.csv](ASL_ISL_MAPPING_REVIEW_TEMPLATE.csv).
The ten rows are a small triage batch chosen from the 126-class experimental
recognizer. Their metrics come from the signer-disjoint held-out test set, but
the support is only 2--6 clips per class. These measurements are useful for
prioritising review; they do not prove a sign is dependable.

For each row, the reviewer must inspect the actual ASL source concept and the
corresponding INCLUDE-50 recording, then fill in:

1. `isl_gloss_confirmed`: the approved ISL gloss/meaning, or `no`.
2. `recorded_motion_confirmed`: whether that exact INCLUDE-50 sequence is an
   appropriate instance of the approved ISL sign, or `no`.
3. `meaning_notes`: sense, regional variation, grammar, non-manual markers, or
   why the English names do not match.
4. reviewer name, date, and `approved`, `rejected`, or `needs_more_data`.

## What happens after approval

For every row marked `approved`, the engineering steps are deterministic:

1. Add the reviewed mapping and reviewer evidence to the playback inventory.
2. Add the alias only for that approved target.
3. Add an end-to-end test proving an ASL result selects the intended recorded
   ISL pose asset rather than the illustrative fallback.
4. Run the full test suite and regenerate the mapping audit.

No row marked `pending`, `rejected`, or `needs_more_data` will be added to the
avatar API. The current three verified paths remain the only enabled
ASL-video-to-recorded-ISL playback routes until this review is complete.
