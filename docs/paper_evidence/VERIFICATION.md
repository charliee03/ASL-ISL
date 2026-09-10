# Handoff verification — 11 September 2026

Executed during this documentation update:

| Check | Result |
| --- | --- |
| `MPLCONFIGDIR=/tmp/aite-mpl .venv/bin/pytest -q` | 38 passed in 4.56 seconds |
| `scripts/export_paper_evidence.py` | Export succeeded; no expected-model missing warning |
| Recognition CSV and acceptance-template parsing | Consistent column counts; six result rows and eight blank acceptance cases |
| New handoff/translation/runbook/evidence-guide relative links | Resolved |
| `main.tex` citation keys | All resolve in `paper_sources.bib` |
| `git diff --check` | Passed |

Not performed: model retraining, fresh full-dataset inference, physical-camera
testing, real alphabet acceptance, paid provider calls, signer review or PDF
compilation. Neither `pdflatex` nor `latexmk` was available on PATH. The existing
`docs/main.pdf` was preserved and is not the updated manuscript.

Build the source on a machine with IEEEtran, BibTeX and a LaTeX installation, or
upload `main.tex` and `paper_sources.bib` to the team's LaTeX editor. From `docs/`,
use a separate output directory to preserve the old PDF:

```bash
mkdir -p /tmp/aite-paper-build
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/aite-paper-build main.tex
```

Inspect the generated PDF manually for overflow, unresolved references and author
details. Confirm the target venue's template; IEEEtran journal mode is only the
current draft format, not a verified venue requirement.

Metric export copies saved summaries; it does not re-establish their experimental
validity. Runtime issues remain in [the handoff](../IEEE_PAPER_HANDOFF.md).
