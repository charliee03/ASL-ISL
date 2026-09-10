# Continuous ISL data acquisition decision

## Decision

Do **not** download the full iSign/ISLTranslate corpus to this laptop now.

The repository's continuous ISL data is the strongest technically appropriate
next resource found during the acquisition review:

- **Resource:** [iSign / ISLTranslate](https://huggingface.co/datasets/Exploration-Lab/iSign)
- **Content:** continuous Indian Sign Language video/pose paired with English
  text; iSign documents SignVideo2Text, SignPose2Text, Text2Pose, word
  prediction, and semantic tasks.
- **License:** Hugging Face lists CC BY-NC-SA 4.0; the ISLTranslate source
  repository documents research/non-commercial CC BY-NC terms. Use is therefore
  research-only and requires the exact access terms accepted at download time.
- **Scale:** Hugging Face reports approximately 228 GB.
- **Access:** gated; the account holder must accept its conditions and share
  contact information before files are available.

This corpus can improve the ISL-side motion and text research, but it is not an
ASL--ISL parallel corpus. It cannot by itself validate an ASL-to-ISL mapping or
train direct ASL-video-to-ISL-pose translation.

## Why it was not downloaded

At the time of this review the local filesystem had approximately 12 GB free.
Downloading 228 GB would fail and would jeopardise the existing project data.
No partial archive is treated as usable training data until its manifest,
checksums, license terms, and split protocol are available.

## Ready next action when storage and access exist

1. Use a project-owned Hugging Face account to accept the iSign conditions.
2. Provide at least 300 GB of external storage or a server-mounted volume.
3. Download the text CSV and a small documented sample first; verify its SHA256
   checksums and UID format.
4. Download pose parts before RGB video when the immediate experiment is
   text-to-pose research; this avoids unnecessary video storage.
5. Use a group-disjoint split by the UID's `video_id` prefix, as specified by
   the dataset authors, to prevent segments from one source video leaking across
   train/validation/test.
6. Keep the corpus outside Git. Record source URL, accepted license version,
   download date, checksums, and the exact split manifest.

## Related public metadata inspected

The public ISLTranslate source repository contains `ISLTranslate.csv` with
31,223 data rows (`uid,text`) and a 291-row signer-validation comparison file.
This confirms that metadata is small, but it does not replace the gated video or
pose data needed for modelling.
