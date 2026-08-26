# Project Attune

Project Attune is a compact, calibrated, time-aligned, uncertainty-aware
paralinguistic transcription layer. It preserves what was said, how it was
audibly expressed, where that expression occurred, and how certain the model is.

Attune is **not a machine that feels emotion**. Its outputs estimate perceived
vocal expression and observable vocal behaviour. They are not verified internal
states, diagnoses, deception judgments, or evidence for high-stakes decisions.

## Research position

The authoritative output is versioned JSON containing a word-timed transcript,
continuous vocal styles, discrete vocal events, affect estimates, and explicit
uncertainty. XML is a deterministic display projection produced only from
validated JSON; a model never generates markup. Spoken text remains separate
from paralinguistic metadata whenever outputs enter a trusted downstream
channel.

The intended base is **SenseVoice-Small (~234M parameters)**, subject to review
of its separate model licence. **Whisper-Small** is the fallback baseline.
Primary deployment is local or low-cost near-real-time inference, with INT8
quantization evaluated later.

## Stage gate

**14-day gate: no large fine-tuning until a baseline report exists.** Phase 1
must first produce `research/baseline-report.md` with transcript, timing,
ontology, calibration, abstention, runtime, and subgroup/error-slice results.
Only then may the project decide whether probes, joint training, or another
approach are justified.

## Quick start

Requires Python 3.12 exactly (the project excludes 3.13) and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest
```

Useful commands:

```bash
make fmt
make lint
make test
uv run python scripts/evaluate.py
```

PyTorch support is intentionally optional in Phase 0:

```bash
uv sync --extra dev --extra torch
```

No model checkpoints are downloaded by installation.

## Repository map

- `src/attune/schema/output.py` — authoritative Pydantic v2 JSON schema (`1.0`)
- `src/attune/schema/xml.py` — deterministic, injection-safe XML renderer
- `src/attune/inference/packaging.py` — separated trusted-channel packaging
- `src/attune/audio/contracts.py` — 16 kHz mono and duration contracts
- `src/attune/evaluation/` — offline metrics and report harness
- `src/attune/baselines/` — lazy baseline adapters and modular cascade
- `docs/` — research, annotation, ontology, ethics, model, and dataset guidance
- `configs/` — staged data/model/training/evaluation/deployment defaults
- `data/fixtures/semantic_conflict/` — synthetic APS/harness wiring fixtures
- `data/provenance/` — dataset licence and provenance ledgers
- `research/` — experiment registry, baseline gate, error analysis, and paper work
- `scripts/` — explicit stage-gated entry-point stubs

Notebooks may be used for exploration only. They are not the training pipeline.
Training and evaluation work belongs in importable `src/attune/` modules with
tested script entry points.

## Output contract

The schema supports quality probabilities, language confidence, word timings,
overlapping styles, between-word events, dimensional and categorical affect,
abstention, and out-of-distribution uncertainty. Affect categories always form
a complete probability distribution. If `affect.abstain` is `true`,
`affect.top_label` **must be `null`**.

See `docs/ontology.md` and `docs/annotation-guide.md` before creating labels.
The required framing is “How does the speaker sound?”, never “What is the
speaker truly feeling?”

## Licence

Code is MIT licensed. Dataset licences and model licences remain independent.
In particular, SenseVoice weights have a separate model licence that must be
reviewed before use or redistribution; the MIT licence does not cover them.
