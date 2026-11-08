import numpy as np
import pytest

from cv.verification.test.verification_experiment import classify
from cv.verification.test.verification_experiment import build_conditions


@pytest.mark.parametrize('predicted,expected', [
    ((0, 0, 10, 10), 'other_region'),
    ((100, 100, 120, 120), 'partial_target'),
    ((0, 0, 500, 500), 'oversized'),
    ((180, 180, 280, 280), 'localization_error'),
])
def test_failure_categories(predicted, expected):
    category, _ = classify(predicted, (100, 100, 200, 200), (1280, 720))
    assert category == expected


def test_conditions_preserve_pairing_and_requested_occlusion():
    background = np.full((720, 1280, 3), 210, np.uint8)
    target = np.zeros((80, 120, 4), np.uint8)
    target[10:70, 10:110] = [20, 80, 130, 255]
    conditions = list(build_conditions(background, target, 42))
    assert len(conditions) == 9
    metadata = [row[4] for row in conditions]
    for key in ('angle', 'side', 'center_x', 'center_y'):
        assert len({row[key] for row in metadata}) == 1
    for size in ('small', 'medium', 'large'):
        group = [row for row in conditions if row[0] == size]
        assert len({row[3] for row in group}) == 1
        for _, ratio, image, answer, meta in group:
            assert image.shape == background.shape and image.dtype == np.uint8
            assert abs(meta['actual_occlusion'] - ratio) < 0.005
            assert 0 <= answer[0] < answer[2] <= 1280
            assert 0 <= answer[1] < answer[3] <= 720


def test_summary_uses_paired_seeds_and_reports_uncertainty(tmp_path):
    import csv
    from cv.verification.test.verification_experiment import METHODS, OCCLUSIONS, SCALES, summarize

    rows = []
    for size in SCALES:
        for ratio in OCCLUSIONS:
            for method in METHODS:
                for seed in (0, 1):
                    correct = seed == 0 and ratio == 0
                    rows.append({'size': size, 'occlusion': ratio, 'method': method,
                                 'seed': seed, 'correct': correct, 'strict_iou50': correct,
                                 'status': 'correct' if correct else 'not_detected', 'seconds': 0.1})
    summary = summarize(rows, tmp_path)
    assert len(summary) == 36
    baseline = next(r for r in summary if r['size'] == 'small' and r['occlusion'] == 0 and r['method'] == 'owlv2')
    assert baseline['accuracy'] == 0.5
    assert baseline['accuracy_ci95_low'] < 0.5 < baseline['accuracy_ci95_high']
    assert baseline['retention_rate'] == 1
    hidden = next(r for r in summary if r['size'] == 'small' and r['occlusion'] == 0.4 and r['method'] == 'owlv2')
    assert hidden['baseline_correct'] == 1 and hidden['retained_correct'] == 0
    assert hidden['retention_rate'] == 0
    assert (tmp_path / 'accuracy_by_conditions.png').exists()
    with (tmp_path / 'summary.csv').open(encoding='utf-8-sig') as file:
        assert len(list(csv.DictReader(file))) == 36
