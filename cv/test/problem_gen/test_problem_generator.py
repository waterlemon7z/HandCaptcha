import numpy as np
import pytest

from cv.problem_gen import ProblemGenerator
from cv.problem_gen.generator import HIT_IOU, NUM_OBJECTS, OUTPUT_SIZE, _iou

from conftest import always_fail, always_solve

W, H = OUTPUT_SIZE
EMPTY = {"correct": False, "answer_box": None, "answer_mask": None}


def _answer_of(gen, problem_id):
    """테스트용: 내부에 저장된 정답 (외부 API로는 판정 전에 볼 수 없음)."""
    return gen._answers[problem_id]


def _shift(box, dx):
    x1, y1, x2, y2 = box
    return x1 + dx, y1, x2 + dx, y2


def test_create_returns_id_image_and_target(asset_dir):
    out = ProblemGenerator(asset_dir).create(seed=1)
    assert len(out) == 3
    problem_id, image, target = out
    assert isinstance(problem_id, str)
    assert image.shape == (H, W, 3) and image.dtype == np.uint8
    assert target.shape == (200, 200, 4)  # 변형 전 원본


def test_image_has_answer_and_distractors(asset_dir):
    # 물체 NUM_OBJECTS개가 합성되고, 그중 하나만 정답 박스와 겹친다
    gen = ProblemGenerator(asset_dir)
    for seed in range(10):
        problem_id, image, _ = gen.create(seed=seed)
        answer = _answer_of(gen, problem_id)
        boxes = always_solve(image, None)
        assert len(boxes) == NUM_OBJECTS, seed
        assert sum(_iou(b, answer.box) >= HIT_IOU for b in boxes) == 1, seed


def test_same_seed_same_problem_new_id(asset_dir):
    gen = ProblemGenerator(asset_dir)
    id1, i1, h1 = gen.create(seed=7)
    id2, i2, h2 = gen.create(seed=7)
    assert id1 != id2
    assert np.array_equal(i1, i2) and np.array_equal(h1, h2)
    assert _answer_of(gen, id1).box == _answer_of(gen, id2).box

    _, i3, _ = gen.create(seed=8)
    assert not np.array_equal(i1, i3)


def test_problem_ids_are_unique(asset_dir):
    gen = ProblemGenerator(asset_dir)
    assert len({gen.create(seed=0)[0] for _ in range(20)}) == 20


def test_answer_inside_frame(asset_dir):
    gen = ProblemGenerator(asset_dir)
    for seed in range(20):
        x1, y1, x2, y2 = _answer_of(gen, gen.create(seed=seed)[0]).box
        assert 0 < x1 < x2 < W and 0 < y1 < y2 < H, seed


def test_solve_exact_box_is_correct_and_reveals(asset_dir):
    gen = ProblemGenerator(asset_dir)
    problem_id, _, _ = gen.create(seed=1)
    answer = _answer_of(gen, problem_id)
    result = gen.solve(problem_id, answer.box)
    assert result["correct"] is True
    assert result["answer_box"] == answer.box
    assert np.array_equal(result["answer_mask"], answer.mask)


def test_solve_accepts_reversed_and_float_box(asset_dir):
    gen = ProblemGenerator(asset_dir)
    problem_id, _, _ = gen.create(seed=1)
    x1, y1, x2, y2 = _answer_of(gen, problem_id).box
    assert gen.solve(problem_id, (x2 + 0.4, y2 - 0.3, x1 - 0.2, y1 + 0.1))["correct"] is True


def test_iou_threshold():
    box = (100, 100, 200, 200)
    # 가로로 dx만큼 밀면 IoU = (100 - dx) / (100 + dx)
    assert _iou(box, _shift(box, 30)) == pytest.approx(70 / 130)   # 0.54 → 정답
    assert _iou(box, _shift(box, 40)) == pytest.approx(60 / 140)   # 0.43 → 오답
    assert _iou(box, (0, 0, 1280, 720)) < HIT_IOU                  # 화면 전체를 덮는 박스는 오답
    assert _iou(box, (150, 150, 150, 300)) == 0.0                  # 넓이 0


def test_solve_wrong_box_still_reveals(asset_dir):
    gen = ProblemGenerator(asset_dir)
    for make in [lambda b: _shift(b, (b[2] - b[0]) // 2 + 5), lambda b: (0, 0, W, H), lambda b: (-50, -50, -10, -10)]:
        problem_id, _, _ = gen.create(seed=1)
        answer = _answer_of(gen, problem_id)
        result = gen.solve(problem_id, make(answer.box))
        assert result["correct"] is False
        assert result["answer_box"] == answer.box


def test_distractor_is_wrong(asset_dir):
    gen = ProblemGenerator(asset_dir)
    problem_id, image, _ = gen.create(seed=1)
    answer_box = _answer_of(gen, problem_id).box
    distractor = min(always_solve(image, None), key=lambda b: _iou(b, answer_box))
    assert gen.solve(problem_id, distractor)["correct"] is False


def test_solved_id_is_discarded(asset_dir):
    gen = ProblemGenerator(asset_dir)
    problem_id, _, _ = gen.create(seed=1)
    box = _answer_of(gen, problem_id).box
    assert gen.solve(problem_id, box)["correct"] is True
    assert gen.solve(problem_id, box) == EMPTY
    assert gen.solve("unknown-id", box) == EMPTY


def test_without_validator_no_history(asset_dir):
    gen = ProblemGenerator(asset_dir)
    gen.create(seed=1)
    assert gen.history == []
    assert gen.auto_solve_rate() is None


def test_validator_receives_image_and_target(asset_dir):
    seen = []

    def spy(image, target):
        seen.append((image, target))
        return []

    _, image, target = ProblemGenerator(asset_dir, validator=spy).create(seed=1)
    [(v_image, v_target)] = seen
    assert np.array_equal(v_image, image) and np.array_equal(v_target, target)


def test_always_fail_validator(asset_dir):
    gen = ProblemGenerator(asset_dir, validator=always_fail)
    problem_id, _, _ = gen.create(seed=1)
    [record] = gen.history
    assert record["problem_id"] == problem_id and record["seed"] == 1
    assert record["attempts"] == [{"boxes": [], "solved": False}]
    assert record["cap_reached"] is False
    assert gen.auto_solve_rate() == 0.0


def test_always_solve_hits_retry_cap(asset_dir):
    gen = ProblemGenerator(asset_dir, validator=always_solve, max_attempts=3)
    problem_id, _, _ = gen.create(seed=1)
    [record] = gen.history
    assert len(record["attempts"]) == 3
    assert all(a["solved"] for a in record["attempts"])
    assert record["cap_reached"] is True
    assert gen.auto_solve_rate() == 1.0
    # 상한에 도달해도 문제는 발급되고 판정할 수 있다
    assert gen.solve(problem_id, _answer_of(gen, problem_id).box)["correct"] is True


def test_distractors_only_is_not_solved(asset_dir):
    # 방해용 물체만 찾은 validator는 못 푼 것으로 본다
    def distractors_only(image, target):
        boxes = always_solve(image, target)
        return sorted(boxes, key=lambda b: max(_iou(b, a) for a in answers))[:-1]

    gen = ProblemGenerator(asset_dir)
    answers = [_answer_of(gen, gen.create(seed=1)[0]).box]
    gen = ProblemGenerator(asset_dir, validator=distractors_only)
    gen.create(seed=1)
    [record] = gen.history
    assert len(record["attempts"][0]["boxes"]) == NUM_OBJECTS - 1
    assert record["attempts"][0]["solved"] is False


def test_regenerates_until_validator_fails(asset_dir):
    calls = []

    def solves_twice(image, target):
        calls.append(image)
        return always_solve(image, target) if len(calls) <= 2 else []

    gen = ProblemGenerator(asset_dir, validator=solves_twice)
    _, image, _ = gen.create(seed=1)
    [record] = gen.history
    assert [a["solved"] for a in record["attempts"]] == [True, True, False]
    assert record["cap_reached"] is False
    assert np.array_equal(image, calls[-1])
    assert not np.array_equal(calls[0], calls[-1])
    assert gen.auto_solve_rate() == 1.0  # 첫 시도에 풀렸으므로


def test_same_seed_same_result_with_validator(asset_dir):
    gen = ProblemGenerator(asset_dir, validator=always_solve, max_attempts=3)
    _, i1, _ = gen.create(seed=5)
    _, i2, _ = gen.create(seed=5)
    assert np.array_equal(i1, i2)


def test_needs_enough_objects(tmp_path):
    (tmp_path / "backgrounds").mkdir()
    (tmp_path / "objects").mkdir()
    with pytest.raises(ValueError):
        ProblemGenerator(tmp_path)
