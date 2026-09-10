# Draft translation: implemented behavior

Updated 11 September 2026; source: `src/translation/translator.py`,
`configs/grammar_rules.json`, `configs/translation.yaml` and `src/api/server.py`.
See [the paper handoff](IEEE_PAPER_HANDOFF.md) for the full evaluation plan.

## Default path

Default behavior is deterministic draft conversion: split tokens, filter configured
fillers, apply direct gloss and special-case mappings, then join the result.
Examples include HELLO → NAMASKAR and WATER → PANI. Hindi transliteration does
not establish ISL grammar. Prose grammar transformations in configuration are
not an implemented tense/classifier/negation system. Context-free filtering can
remove meaningful LIKE, RIGHT, SO or WELL. Unknown tokens may have no motion.

`/translate` returns `asl_gloss`, `isl_gloss`, `translation_mode` and
`review_required`. AI triage in the review CSV is not signer approval; the runtime
does not enforce that CSV as a linguistic-approval gate.

## Optional providers

| Setting | Purpose | Default |
| --- | --- | --- |
| `AITE_ENABLE_GEMINI` | Remote draft refinement | false |
| `GEMINI_API_KEY` | Server-side credential | kept outside repository |
| `AITE_GEMINI_MODEL` | Requested model | `gemini-2.5-flash` in current code |
| `AITE_ENABLE_LLM` | Local Hugging Face Llama path | false |

Gemini receives filtered text, draft and vocabulary, not uploaded video in this
integration. The vocabulary comes from mappings and aliases, not exclusively
reviewed/asset-backed signs. `isl_glosses` is checked against allowed tokens;
optional `unsupported_source_tokens` is appended after that check without
equivalent source-membership/completeness validation. Format checks do not
guarantee meaning preservation. No explicit timeout/output bound is configured
in this layer. Provider errors normally fall back; with both flags enabled,
Gemini failure may proceed to Llama. Exception logging deserves privacy review.

Modes include `draft_rule_based`, `gemini_constrained_draft` and `llama_draft`.
None is a correctness certificate. Llama-2-7b-chat is optional and has not been
trained here on verified parallel ASL–ISL data. Four-bit loading is CUDA-dependent;
do not claim a universal RAM requirement or measured latency.

Load secrets only in the server terminal when intentionally testing the provider:

```bash
set -a
source "$HOME/.config/aite/secrets.env"
set +a
AITE_ENABLE_GEMINI=true AITE_ENABLE_LLM=false \
  .venv/bin/uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

A populated key does not prove provider access. Rotate the credential previously
shared in chat; never include it in source, paper, screenshots or evidence.
Do not send private input without authorization.

## Output and evidence boundaries

The renderer is a separate retrieval/fallback component. It may render only the
first recognized word in a multiword input; approximate sentence matches may
change meaning. Fingerspelling requires real alphabet assets and may skip missing
letters. A successful MP4 response does not establish complete ISL translation.

Tests cover selected behavior and mocked provider responses, not human adequacy.
No verified parallel evaluation set or measured Gemini-versus-rules improvement
is established. For a future comparison, freeze unseen inputs, compare the same
cases across modes, record model/version, actual mode, failures, latency and cost,
and ask competent ISL reviewers to score meaning, negation, names, completeness
and intelligibility separately. Include failures and reviewer disagreement.

Earlier speculative claims are retained in [TRANSLATION_LEGACY.md](TRANSLATION_LEGACY.md),
not endorsed as results.
