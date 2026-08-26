"""Phase 1 baseline runners and modular cascade."""

from attune.baselines.adapters import (
    BaselineInput,
    BaselineUnavailableError,
    Emotion2VecPlusAdapter,
    SenseVoiceSmallAdapter,
    TranscriptSentimentAdapter,
    WhisperSmallAdapter,
)
from attune.baselines.cascade import ModularCascade

__all__ = [
    "BaselineInput",
    "BaselineUnavailableError",
    "Emotion2VecPlusAdapter",
    "ModularCascade",
    "SenseVoiceSmallAdapter",
    "TranscriptSentimentAdapter",
    "WhisperSmallAdapter",
]
