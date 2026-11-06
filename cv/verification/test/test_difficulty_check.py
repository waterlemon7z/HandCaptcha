from unittest.mock import patch

import pytest

from cv.verification.difficulty_check import DifficultyCheck


def test_owlv2_loads_on_selection_and_is_reused():
    with patch("cv.verification.difficulty_check.Owlv2Checker") as model:
        checker = DifficultyCheck()
        model.assert_not_called()
        model.return_value.detect.return_value = [(1, 2, 30, 40)]
        image, target = object(), object()
        for _ in range(2):
            assert checker.difficult_check(image, target, method="owlv2") == [(1, 2, 30, 40)]
        model.assert_called_once_with()
        model.return_value.detect.assert_called_with(image, target)


@pytest.mark.parametrize("method", ["auto", "orb", "sift"])
def test_feature_methods_do_not_load_owlv2(method):
    with patch("cv.verification.difficulty_check.Owlv2Checker") as model:
        checker = DifficultyCheck()
        with patch.object(checker._orb_sift, "detect", return_value=[]) as detect:
            image, target = object(), object()
            assert checker.difficult_check(image, target, method=method) == []
            detect.assert_called_once_with(image, target, method=method)
        model.assert_not_called()


@pytest.mark.parametrize("method", ["", "owl", "2"])
def test_method_must_match_full_name(method):
    with patch("cv.verification.difficulty_check.Owlv2Checker") as model:
        with pytest.raises(ValueError):
            DifficultyCheck().difficult_check(None, None, method=method)
        model.assert_not_called()
