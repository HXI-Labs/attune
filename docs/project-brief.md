# Project brief

## Thesis

Attune tests whether one compact, calibrated layer can jointly preserve lexical
content, vocal behaviour, timing, and uncertainty well enough for local or
low-cost near-real-time use. It describes perceptible acoustics; it does not
infer privileged access to a person's mind.

## Rejected formulation

Attune is not “ASR plus angry/sad XML tags.” That formulation collapses
time-varying behaviour into one uncalibrated judgment, encourages a model to
generate unsafe markup, loses overlap and event placement, and presents a
subjective interpretation as fact. JSON is authoritative; deterministic XML is
only a projection.

## Research questions

1. Does a compact shared representation improve vocal-style and event detection
   without harming transcription or timing?
2. Which observable labels are reliable across annotators, speakers, languages,
   recording conditions, and short utterances?
3. Can calibration, out-of-distribution scoring, and abstention make subjective
   affect outputs appropriately cautious?
4. What accuracy/latency trade-off is attainable locally, and what is lost
   under later INT8 quantization?

## Stages and gates

- **Phase 0 — contract:** schema, ontology, ethics, provenance, safe renderers,
  test scaffolding, and explicit stage-gated scripts.
- **Phase 1 — baselines:** evaluate SenseVoice-Small (~234M; licence review
  required) and Whisper-Small fallback on speaker-disjoint data. Publish the
  baseline report.
- **14-day gate:** no large fine-tuning until the baseline report covers ASR,
  alignment, behaviour labels, calibration, abstention, OOD slices, runtime,
  and errors.
- **Phase 2 — probes:** train small frozen-encoder probes only if baselines
  establish viable labels and data.
- **Phase 3 — joint training:** proceed only with evidence that a joint model is
  necessary and ethically supportable.
- **Phase 4 — deployment:** evaluate streaming, ONNX, and INT8 without weakening
  uncertainty semantics or channel separation.

Every stage must record code/data/model versions, licences, splits, seeds,
metrics, calibration, and known failure modes in the experiment registry.
