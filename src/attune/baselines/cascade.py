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
from attune.calibration import load_affect_abstention, load_calibration
from attune.evaluation.report import RuntimeMetrics
from attune.models.probe_inference import (
    FSD50K_LABEL_MAPPING,
    VOCALSOUND_LABEL_MAPPING,
    FrozenEncoderProvider,
    FrozenLinearProbeHead,
    ProbePrediction,
)
from attune.models.temporal_probe import FrozenTemporalProbeHead
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

        # The ASR output is authoritative: genuine returned alignment passes
        # through unchanged, while unavailable alignment remains an empty list.
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
                            "start_ms": annotation.start_ms,
                            "end_ms": annotation.end_ms,
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
                    "crying_speech. A held-out-gated temporal source replaces utterance "
                    "scope for the same event label and may emit multiple spans. All spans "
                    "are provisional."
                ),
                "timestamp_policy": (
                    "DCASE-overlap events use gated frame spans when configured; other "
                    "event/style spans remain utterance-level. Genuine ASR word alignment "
                    "passes through when available and is never inferred."
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
        calibration_path: Path | None = None,
        temporal_head_checkpoints: tuple[Path, ...] = (),
    ) -> None:
        vocalsound_calibration = (
            load_calibration(calibration_path, component="vocalsound_probe")
            if calibration_path is not None
            else None
        )
        fsd50k_calibration = (
            load_calibration(calibration_path, component="fsd50k_probe")
            if calibration_path is not None
            else None
        )
        encoder = FrozenEncoderProvider(sensevoice_checkpoint, embedding_cache)
        heads: tuple[EventStyleHead, ...] = (
            FrozenLinearProbeHead(
                name="vocalsound-frozen-linear-probe",
                checkpoint=vocalsound_probe_checkpoint,
                encoder=encoder,
                label_mapping=VOCALSOUND_LABEL_MAPPING,
                calibration=vocalsound_calibration,
            ),
            FrozenLinearProbeHead(
                name="fsd50k-frozen-linear-probe",
                checkpoint=fsd50k_probe_checkpoint,
                encoder=encoder,
                label_mapping=FSD50K_LABEL_MAPPING,
                calibration=fsd50k_calibration,
            ),
        )
        if temporal_head_checkpoints:
            heads = (
                *heads,
                *(
                    FrozenTemporalProbeHead(
                        checkpoint=checkpoint,
                        sensevoice_checkpoint=sensevoice_checkpoint,
                        frame_cache=embedding_cache,
                    )
                    for checkpoint in temporal_head_checkpoints
                ),
            )
        super().__init__(
            asr=SenseVoiceSmallAdapter(checkpoint=sensevoice_checkpoint),
            affect=Emotion2VecPlusAdapter(
                checkpoint=emotion2vec_checkpoint,
                calibration=(
                    load_calibration(
                        calibration_path,
                        component="emotion2vec_plus_affect",
                    )
                    if calibration_path is not None
                    else None
                ),
                abstention=(
                    load_affect_abstention(
                        calibration_path,
                        component="emotion2vec_plus_affect",
                    )
                    if calibration_path is not None
                    else None
                ),
            ),
            event_heads=heads,
        )
        temporal_suffix = (
            f"+{len(temporal_head_checkpoints)}-gated-frame-heads"
            if temporal_head_checkpoints
            else ""
        )
        self.name = (
            "attune-cascade:sensevoice+emotion2vec+aed+vocalsound-probe+fsd50k-probe"
            f"{temporal_suffix}"
        )


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
    """Merge labels while preferring gated frame spans over utterance scope."""
    selected: dict[tuple[str, str], list[tuple[dict[str, object], str]]] = {}
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
            start_ms = annotation.get("start_ms")
            end_ms = annotation.get("end_ms")
            if (start_ms is None) != (end_ms is None):
                raise ValueError("timed annotations require both start_ms and end_ms")
            timed = start_ms is not None
            existing = selected.get(key)
            if existing is None:
                selected[key] = [(annotation, source)]
                continue
            existing_timed = existing[0][0].get("start_ms") is not None
            if timed and not existing_timed:
                decisions.append(
                    {
                        "channel": channel,
                        "label": label,
                        "kept_source": source,
                        "replaced_utterance_source": existing[0][1],
                    }
                )
                selected[key] = [(annotation, source)]
                continue
            if timed and existing_timed and existing[0][1] == source:
                existing.append((annotation, source))
                continue
            decisions.append(
                {
                    "channel": channel,
                    "label": label,
                    "kept_source": existing[0][1],
                    "suppressed_duplicate_source": source,
                }
            )

    events: list[dict[str, object]] = []
    styles: list[dict[str, object]] = []
    for (channel, label), annotations in selected.items():
        for annotation, _source in annotations:
            confidence = float(annotation["confidence"])
            start_ms = annotation.get("start_ms")
            end_ms = annotation.get("end_ms")
            if channel == "event":
                EventLabel(label)
                events.append(
                    {
                        "id": f"e{len(events) + 1}",
                        "label": label,
                        "start_ms": 0 if start_ms is None else int(start_ms),
                        "end_ms": duration_ms if end_ms is None else int(end_ms),
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
                        "start_ms": 0 if start_ms is None else int(start_ms),
                        "end_ms": duration_ms if end_ms is None else int(end_ms),
                        "start_word_id": None,
                        "end_word_id": None,
                        "confidence": confidence,
                        "status": Status.PROVISIONAL.value,
                    }
                )
            else:
                raise ValueError(f"unsupported annotation channel: {channel}")
    return events, styles, decisions
