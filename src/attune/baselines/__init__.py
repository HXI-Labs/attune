"""Baseline runners and the modular cascade."""

from attune.baselines.adapters import (
    BaselineInput,
    BaselineUnavailableError,
    Emotion2VecPlusAdapter,
    SenseVoiceSmallAdapter,
    TranscriptSentimentAdapter,
    WhisperSmallAdapter,
)
from attune.baselines.cascade import AttuneCascade, ModularCascade

__all__ = [
    "BaselineInput",
    "BaselineUnavailableError",
    "AttuneCascade",
    "Emotion2VecPlusAdapter",
    "ModularCascade",
    "SenseVoiceSmallAdapter",
    "TranscriptSentimentAdapter",
    "WhisperSmallAdapter",
]
