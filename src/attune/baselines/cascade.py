"""Phase 1 modular cascade: ASR + affect + optional event-head override."""

from __future__ import annotations

from dataclasses import dataclass

from attune.baselines.adapters import (
    BaselineAdapter,
    BaselineInput,
    BaselinePrediction,
)
from attune.evaluation.report import RuntimeMetrics
from attune.schema.output import AttuneOutput


@dataclass(frozen=True)
class StubEventHead:
    """No-op event component used until an evaluated event model is selected."""

    name: str = "stub-event-head"

    def predict(self) -> list[object]:
        return []


class ModularCascade(BaselineAdapter):
    """Compose independently replaceable runners without training new weights."""

    def __init__(
        self,
        *,
        asr: BaselineAdapter,
        affect: BaselineAdapter,
        event_head: StubEventHead | None = None,
    ) -> None:
        self.asr = asr
        self.affect = affect
        self.event_head = event_head
        event_source_name = event_head.name if event_head else "asr-event-output"
        self.name = f"cascade:{asr.name}+{affect.name}+{event_source_name}"

    def availability(self) -> tuple[bool, str | None]:
        for component in (self.asr, self.affect):
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
        payload["model"] = {"name": self.name, "version": "phase-1", "quantization": None}
        payload["affect"] = affect_prediction.output.affect.model_dump(mode="json")
        if self.event_head is not None:
            payload["events"] = self.event_head.predict()
        output = AttuneOutput.model_validate(payload)
        elapsed = asr_prediction.runtime.elapsed_seconds + affect_prediction.runtime.elapsed_seconds
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
                "note": (
                    "Cascade output uses the acoustic affect component; "
                    "the ASR adapter's placeholder affect is replaced."
                ),
            },
        )
