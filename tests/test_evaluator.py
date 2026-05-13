import numpy as np

from sam3_uq.backends.mock import MockSamBackend
from sam3_uq.evaluator import Sam3UQEvaluator


def test_evaluator_returns_expected_outputs():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:20, 10:22] = 1

    evaluator = Sam3UQEvaluator(MockSamBackend(mask), min_component_area=4)
    result = evaluator.evaluate(image=image, model_mask=mask, concept="lesion")

    assert "u_image" in result.scores
    assert 0.0 <= result.scores["u_image"] <= 1.0
    assert result.pixel_uncertainty.shape == mask.shape
    assert result.consensus_mask.shape == mask.shape
    assert len(result.instance_scores) == 1
