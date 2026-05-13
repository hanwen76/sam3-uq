import numpy as np

from sam3_uq.metrics import boundary_disagreement, dice, iou, prompt_instability


def test_dice_and_iou_identical_empty_masks():
    a = np.zeros((4, 4), dtype=bool)
    assert dice(a, a) == 1.0
    assert iou(a, a) == 1.0


def test_dice_and_iou_partial_overlap():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[:2, :2] = True
    b[1:3, 1:3] = True
    assert round(dice(a, b), 4) == 0.25
    assert round(iou(a, b), 4) == 0.1429


def test_prompt_instability_zero_for_same_masks():
    a = np.zeros((8, 8), dtype=bool)
    a[2:6, 2:6] = True
    assert prompt_instability([a, a.copy()]) < 1e-6


def test_boundary_disagreement_increases_for_shift():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[2:6, 2:6] = True
    b[2:6, 3:7] = True
    assert boundary_disagreement(a, b) > 0.0
