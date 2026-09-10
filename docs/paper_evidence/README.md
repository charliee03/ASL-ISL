# Paper evidence snapshot

See [VERIFICATION.md](VERIFICATION.md) for checks actually run and PDF build status.

Run `.venv/bin/python scripts/export_paper_evidence.py` from the repository root.
The exporter reads a fixed selection of local experiment files and writes:

- `metrics.json`: selected scalar metrics from saved results (no raw predictions or input paths).
- `recognition_results.csv`: manuscript-friendly recognition table, values as fractions.
- `manifest.json`: SHA-256 hashes and sizes for model/config/split/source files, Git revision and dirty state, and missing expected artifacts.

The export does not train, call a provider, access secrets or copy datasets/checkpoints. It is an inventory of current files, not independent verification of experiment quality. Keep a private full artifact backup as described in [the handoff](../IEEE_PAPER_HANDOFF.md). Re-export after changes; hashes are expected to change. Generated files overwrite only these three outputs.

The manifest intentionally excludes raw video, secrets, logs, virtual environments and arbitrary home-directory contents. Inspect even sanitized artifacts before public release. Missing files are reported rather than invented.
