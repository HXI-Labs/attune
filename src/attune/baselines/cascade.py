"""Attune cascade: ASR, acoustic affect, AED, and frozen linear probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from attune.baselines.adapters import (
    BaselineAdapter,
    BaselineInput,
    BaselinePrediction,
    Emotion2VecPlusAdapter,
    SenseVoiceSmallAdapter,
)
from attune.evaluation.report import RuntimeMetrics
from attune.models.probe_inference import (
    FSD50K_LABEL_MAPPING,
    VOCALSOUND_LABEL_MAPPING,
    FrozenEncoderProvider,
    FrozenLinearProbeHead,
    ProbePrediction,
)
from attune.schema.output import AttuneOutput, EventLabel, Status, StyleLabel


@dataclass(frozen=True)
class StubEventHead:
    """No-op event component retained for explicit fixture wiring."""

    name: str = "stub-event-head"

    def availability(self) -> tuple[bool, str | None]:
        return True, None

    def predict(self, audio_path: Path) -> ProbePrediction:
        del audio_path
        return ProbePrediction(annotations=(), elapsed_seconds=0.0, diagnostics={})


class EventStyleHead(Protocol):
    """Minimal interface implemented by utterance-level probe heads."""

    name: str

    def availability(self) -> tuple[bool, str | None]: ...

    def predict(self, audio_path: Path) -> ProbePrediction: ...


class ModularCascade(BaselineAdapter):
    """Compose replaceable runners and merge utterance-level event/style evidence."""

    def __init__(
        self,
        *,
        asr: BaselineAdapter,
        affect: BaselineAdapter,
        event_heads: tuple[EventStyleHead, ...] = (),
        event_head: EventStyleHead | None = None,
    ) -> None:
        if event_head is not None and event_heads:
            raise ValueError("use event_head or event_heads, not both")
        self.asr = asr
        self.affect = affect
        self.event_heads = event_heads or ((event_head,) if event_head is not None else ())
        event_source_names = ["asr-aed", *(head.name for head in self.event_heads)]
        self.name = f"cascade:{asr.name}+{affect.name}+{'+'.join(event_source_names)}"

    def availability(self) -> tuple[bool, str | None]:
        for component in (self.asr, self.affect, *self.event_heads):
            available, reason = component.availability()
            if not available:
                return False, f"{component.name}: {reason}"
        return True, None

    def predict(self, item: BaselineInput) -> BaselinePrediction:
        asr_prediction = self.asr.predict(item)
        affect_input = BaselineInput(
            audio_path=item.audio_path,
            transcript_hint=asr_prediction.output.transcript.text,
            language_hint=asr_prediction.output.language.label,
        )
        affect_prediction = self.affect.predict(affect_input)
        payload = asr_prediction.output.model_dump(mode="json")
        payload["model"] = {"name": self.name, "version": "cascade-1", "quantization": None}
        payload["affect"] = affect_prediction.output.affect.model_dump(mode="json")

        # ASR adapters expose no genuine alignment. The cascade therefore keeps
        # transcript text but emits no invented word timestamps or word anchors.
        payload["transcript"]["words"] = []
        components = [_asr_annotations(asr_prediction.output, self.asr.name)]
        elapsed = asr_prediction.runtime.elapsed_seconds + affect_prediction.runtime.elapsed_seconds
        probe_diagnostics = []
        for head in self.event_heads:
            prediction = head.predict(item.audio_path)
            elapsed += prediction.elapsed_seconds
            components.append(
                {
                    "name": head.name,
                    "abstained": prediction.abstained,
                    "annotations": [
                        {
                            "channel": annotation.channel,
                            "label": annotation.label.value,
                            "confidence": annotation.confidence,
                        }
                        for annotation in prediction.annotations
                    ],
                }
            )
            probe_diagnostics.append(prediction.diagnostics)

        events, styles, decisions = _merge_annotations(
            components,
            duration_ms=asr_prediction.output.audio.duration_ms,
        )
        payload["events"] = events
        payload["styles"] = styles
        output = AttuneOutput.model_validate(payload)
        return BaselinePrediction(
            output=output,
            runtime=RuntimeMetrics.measured(
                audio_seconds=output.audio.duration_ms / 1000,
                elapsed_seconds=elapsed,
            ),
            diagnostics={
                **(affect_prediction.diagnostics or {}),
                "asr_component": {
                    "name": self.asr.name,
                },
                "affect_component": {
                    "name": self.affect.name,
                },
                "event_style_components": components,
                "probe_diagnostics": probe_diagnostics,
                "merge_decisions": decisions,
                "merge_policy": (
                    "Set union by channel and ontology label. ASR/SenseVoice AED is "
                    "considered first; an abstaining probe contributes nothing, so AED-only "
                    "annotations remain unchanged. If AED and both probes are empty, no "
                    "event/style is emitted. Non-abstaining probes fill missing labels and "
                    "duplicate labels are suppressed. Scream is a discrete event and never "
                    "becomes shouting. Sob is a discrete event and never becomes "
                    "crying_speech. All spans cover the utterance and are provisional."
                ),
                "timestamp_policy": (
                    "utterance-level only; no word alignment or frame localization inferred"
                ),
                "note": (
                    "Cascade output uses emotion2vec+ acoustic affect; transcript lexicon "
                    "and the ASR adapter's placeholder affect are not used."
                ),
            },
        )


class AttuneCascade(ModularCascade):
    """The inspected Attune cascade built entirely from local reviewed artifacts."""

    def __init__(
        self,
        *,
        sensevoice_checkpoint: Path,
        emotion2vec_checkpoint: Path,
        vocalsound_probe_checkpoint: Path,
        fsd50k_probe_checkpoint: Path,
        embedding_cache: Path,
    ) -> None:
        encoder = FrozenEncoderProvider(sensevoice_checkpoint, embedding_cache)
        heads = (
            FrozenLinearProbeHead(
                name="vocalsound-frozen-linear-probe",
                checkpoint=vocalsound_probe_checkpoint,
                encoder=encoder,
                label_mapping=VOCALSOUND_LABEL_MAPPING,
            ),
            FrozenLinearProbeHead(
                name="fsd50k-frozen-linear-probe",
                checkpoint=fsd50k_probe_checkpoint,
                encoder=encoder,
                label_mapping=FSD50K_LABEL_MAPPING,
            ),
        )
        super().__init__(
            asr=SenseVoiceSmallAdapter(checkpoint=sensevoice_checkpoint),
            affect=Emotion2VecPlusAdapter(checkpoint=emotion2vec_checkpoint),
            event_heads=heads,
        )
        self.name = "attune-cascade:sensevoice+emotion2vec+aed+vocalsound-probe+fsd50k-probe"


def _asr_annotations(output: AttuneOutput, source_name: str) -> dict[str, object]:
    return {
        "name": f"{source_name}-aed",
        "annotations": [
            {
                "channel": "event",
                "label": event.label.value,
                "confidence": event.confidence,
            }
            for event in output.events
        ]
        + [
            {
                "channel": "style",
                "label": style.label.value,
                "confidence": style.confidence,
            }
            for style in output.styles
        ],
    }


def _merge_annotations(
    components: list[dict[str, object]],
    *,
    duration_ms: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    """Merge component sets in precedence order and materialize whole-clip spans."""
    selected: dict[tuple[str, str], tuple[float, str]] = {}
    decisions: list[dict[str, str]] = []
    for component in components:
        source = str(component["name"])
        annotations = component["annotations"]
        if not isinstance(annotations, list):
            raise TypeError("event/style component annotations must be a list")
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise TypeError("event/style annotations must be dictionaries")
            channel = str(annotation["channel"])
            label = str(annotation["label"])
            key = (channel, label)
            if key in selected:
                decisions.append(
                    {
                        "channel": channel,
                        "label": label,
                        "kept_source": selected[key][1],
                        "suppressed_duplicate_source": source,
                    }
                )
                continue
            selected[key] = (float(annotation["confidence"]), source)

    events: list[dict[str, object]] = []
    styles: list[dict[str, object]] = []
    for (channel, label), (confidence, _source) in selected.items():
        if channel == "event":
            EventLabel(label)
            events.append(
                {
                    "id": f"e{len(events) + 1}",
                    "label": label,
                    "start_ms": 0,
                    "end_ms": duration_ms,
                    "after_word_id": None,
                    "confidence": confidence,
                    "status": Status.PROVISIONAL.value,
                }
            )
        elif channel == "style":
            StyleLabel(label)
            styles.append(
                {
                    "id": f"s{len(styles) + 1}",
                    "label": label,
                    "start_ms": 0,
                    "end_ms": duration_ms,
                    "start_word_id": None,
                    "end_word_id": None,
                    "confidence": confidence,
                    "status": Status.PROVISIONAL.value,
                }
            )
        else:
            raise ValueError(f"unsupported annotation channel: {channel}")
    return events, styles, decisions
