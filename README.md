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

The intended base is **SenseVoice-Small (~234M parameters)**, with
**Whisper-Small** as the fallback baseline and emotion2vec+ as an affect
baseline. Their 2026-08-26 licence review permits downloading official weights
for internal baseline runs only. Primary deployment is local or low-cost
near-real-time inference, with INT8 quantization evaluated later.

## Stage gate

Phase 1 closure is tracked in `research/phase1-close.md`. The gold gate remains
closed: current event/style targets are weak source labels and their existing
spans are whole utterances, not localization. Third-party model/data licences
remain independent, public weight redistribution is not authorized, and
MSP-Podcast use remains pending review.

Phase 2 is the frozen-encoder probe package in `research/phase2-probes.md`.
SenseVoice-Small remains fully frozen; only small linear probe heads were
trained. The package adds validation-selected affect abstention and a
weight-refusing local WAV CLI, but does not authorize gold claims or Phase 3
joint training. The gate remains closed.

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

PyTorch support is optional and required only for local model runners and the
Stage 2 frozen event probe:

```bash
uv sync --extra dev --extra torch --extra model-runners
```

No model checkpoints are downloaded by installation.

`AttuneCascade` is the concrete inspected runner: SenseVoiceSmall transcript
and AED, emotion2vec+ affect, and the union of the frozen VocalSound and FSD50K
linear probes when their validation-selected max-softmax, energy, or
genuine-negative `none`-logit checks do not abstain. Abstention contributes no
event/style, preserving AED-only output. Run the combined original-150 plus
licence-clean-160 inspection with
`scripts/evaluate_attune_cascade.py` after preparing the bounded datasets,
reviewed local model paths, and gitignored heads. The encoder remains frozen,
all annotations remain provisional, and the scientific gold gate is closed.
The frame-level DCASE retry scores 0.7285 segment / 0.4637 collar F1 directly;
validation-selected hysteresis trades segment F1 to 0.7059 while improving
collar F1 to 0.5279, versus 0.3183 / 0 for whole-clip. It enables gated frame
spans for laugh/cough/throat-clear when its gitignored checkpoint is configured.
Other `0..duration` event/style spans remain utterance scope, never
localization. A separate natural-scene STARSS23 laughter head on 10 s crops scores 0.7381
segment F1 versus 0.4894 whole-clip, but only 0.1159 collar F1. A 60 s scene
raster MLP scores 0.4794 versus 0.1721 whole-clip and 0.1074 collar F1; the
segment margin clears +0.05 but collar F1 fails the fixed 0.25 gate, so
STARSS23 timestamps stay unwired. Details are in `research/timing-holes.md`.
For one or more local WAV files, `scripts/infer.py` emits authoritative JSON
and optional deterministic XML; exact offline commands are in
`docs/baseline-runners.md`.

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
- `scripts/` — explicit stage-gated entry points, including the frozen event probe

Notebooks may be used for exploration only. They are not the training pipeline.
Training and evaluation work belongs in importable `src/attune/` modules with
tested script entry points.

## Output contract

The schema supports quality probabilities, language confidence, genuine word timings,
overlapping styles, between-word events, dimensional and categorical affect,
abstention, and out-of-distribution uncertainty. Affect categories always form
a complete probability distribution. If `affect.abstain` is `true`,
`affect.top_label` **must be `null`**.

SenseVoice and Whisper adapters populate `transcript.words` only from explicit
model-returned alignment. Missing or invalid alignment remains `[]`; Attune
never interpolates words across an utterance.

See `docs/ontology.md` and `docs/annotation-guide.md` before creating labels.
The required framing is “How does the speaker sound?”, never “What is the
speaker truly feeling?”

## Licence

Code is MIT licensed. Dataset licences and model licences remain independent.
Official SenseVoice-Small weights use the
[FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE);
use requires attribution to FunASR/FunAudioLLM SenseVoiceSmall, retention of the
model name, and a link to that licence. Third-party conversions require their
own review. Whisper's upstream project licenses its code and original weights
under MIT, while its Hugging Face card currently says Apache-2.0; this project
prefers the upstream MIT licence. emotion2vec+ weight cards use `model-license`
from the FunASR model-agreement family; their weights are not covered by the
emotion2vec code licence. Exact review records are in `data/provenance/`.
