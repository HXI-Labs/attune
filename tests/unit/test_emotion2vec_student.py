from pathlib import Path

import pytest

from attune.inference.emotion2vec_student import TruncatedEmotion2VecPredictor


def test_predictor_rejects_unknown_quantization_before_loading_assets() -> None:
    with pytest.raises(ValueError, match="quantization must be fp32 or int8"):
        TruncatedEmotion2VecPredictor(
            Path("missing-teacher"),
            Path("missing-checkpoint"),
            quantization="int4",
        )
