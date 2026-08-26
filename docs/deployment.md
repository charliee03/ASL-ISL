# Deployment and defense handoff

## Local run

```bash
.venv/bin/uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. The service uses the included recognition checkpoint and a deterministic local translation fallback by default; set `AITE_ENABLE_LLM=true` only when the required Hugging Face model is available and authorised.

## Docker

```bash
docker build -t aite:local .
docker run --rm -p 8000:8000 aite:local
```

Then open `http://localhost:8000/health` and the UI at `http://localhost:8000`.

The image includes the source, static UI, configs, and recognition checkpoint. It intentionally excludes the training dataset and local virtual environments.

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
