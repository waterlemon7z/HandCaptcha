import cv2
import numpy as np
import pytest

from cv.problem_gen import ProblemGenerator
from cv.problem_gen.generator import HIT_BOX_PX, NUM_OBJECTS, OUTPUT_SIZE, _hit

from conftest import always_fail, always_solve

W, H = OUTPUT_SIZE
DONE_EMPTY = {"correct": False, "done": True, "found": 0, "box": None, "answer_boxes": None, "answer_masks": None}


def _answer_of(gen, problem_id):
    """테스트용: 내부에 저장된 정답 (외부 API로는 문제가 끝나기 전에 볼 수 없음)."""
    return gen._answers[problem_id]


def _inside_point(mask):
    # 마스크 안에서 경계와 가장 먼 점
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(x), int(y)


def _far_point(answer):
    # 어느 물체와도 먼 점
    dist = cv2.distanceTransform(255 - np.maximum.reduce(answer.masks), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(x), int(y)


def test_create_returns_id_image_and_hints(asset_dir):
    out = ProblemGenerator(asset_dir).create(seed=1)
    assert len(out) == 3
    problem_id, image, hints = out
    assert isinstance(problem_id, str)
    assert image.shape == (H, W, 3) and image.dtype == np.uint8
    assert len(hints) == NUM_OBJECTS
    assert all(h.ndim == 3 and h.shape[2] == 4 for h in hints)


def test_hints_are_distinct_originals(asset_dir):
    _, _, hints = ProblemGenerator(asset_dir).create(seed=1)
    assert all(h.shape == (200, 200, 4) for h in hints)  # 변형 전 원본 크기
    assert len({h.tobytes() for h in hints}) == NUM_OBJECTS


def test_same_seed_same_problem_new_id(asset_dir):
    gen = ProblemGenerator(asset_dir)
    id1, i1, h1 = gen.create(seed=7)
    id2, i2, h2 = gen.create(seed=7)
    assert id1 != id2
    assert np.array_equal(i1, i2)
    assert all(np.array_equal(a, b) for a, b in zip(h1, h2))
    assert _answer_of(gen, id1).boxes == _answer_of(gen, id2).boxes

    _, i3, _ = gen.create(seed=8)
    assert not np.array_equal(i1, i3)


def test_problem_ids_are_unique(asset_dir):
    gen = ProblemGenerator(asset_dir)
    assert len({gen.create(seed=0)[0] for _ in range(20)}) == 20


def test_objects_inside_frame_and_apart(asset_dir):
    gen = ProblemGenerator(asset_dir)
    for seed in range(20):
        answer = _answer_of(gen, gen.create(seed=seed)[0])
        assert len(answer.masks) == NUM_OBJECTS
        for x1, y1, x2, y2 in answer.boxes:
            assert 0 < x1 < x2 < W and 0 < y1 < y2 < H, seed
        overlap = np.sum([m > 0 for m in answer.masks], axis=0)
        assert overlap.max() == 1, seed


def test_find_all_in_any_order(asset_dir):
    gen = ProblemGenerator(asset_dir)
    problem_id, _, _ = gen.create(seed=1)
    answer = _answer_of(gen, problem_id)
    masks, boxes = list(answer.masks), list(answer.boxes)
    for n, i in enumerate([2, 0, 1], start=1):
        result = gen.solve(problem_id, _inside_point(masks[i]))
        assert result["correct"] is True and result["found"] == n
        assert result["box"] == boxes[i]
        assert result["done"] is (n == NUM_OBJECTS)
    assert result["answer_boxes"] == boxes
    assert all(np.array_equal(a, b) for a, b in zip(result["answer_masks"], masks))
    assert gen.solve(problem_id, _inside_point(masks[0])) == DONE_EMPTY


def test_answers_hidden_until_done(asset_dir):
    gen = ProblemGenerator(asset_dir)
    problem_id, _, _ = gen.create(seed=1)
    result = gen.solve(problem_id, _inside_point(_answer_of(gen, problem_id).masks[0]))
    assert result["done"] is False
    assert result["answer_boxes"] is None and result["answer_masks"] is None


def test_duplicate_hit_is_ignored(asset_dir):
    gen = ProblemGenerator(asset_dir)
    problem_id, _, _ = gen.create(seed=1)
    point = _inside_point(_answer_of(gen, problem_id).masks[0])
    gen.solve(problem_id, point)
    result = gen.solve(problem_id, point)
    assert result["correct"] is False and result["done"] is False and result["found"] == 1
    assert problem_id in gen._answers


def test_miss_ends_problem_and_reveals(asset_dir):
    gen = ProblemGenerator(asset_dir)
    for point in [None, (-5, 10), (W + 3, H // 2)]:
        problem_id, _, _ = gen.create(seed=1)
        answer = _answer_of(gen, problem_id)
        boxes = list(answer.boxes)
        result = gen.solve(problem_id, _far_point(answer) if point is None else point)
        assert result["correct"] is False and result["done"] is True and result["found"] == 0
        assert result["answer_boxes"] == boxes
        assert gen.solve(problem_id, _inside_point(answer.masks[0])) == DONE_EMPTY


def test_unknown_id(asset_dir):
    assert ProblemGenerator(asset_dir).solve("unknown-id", (10, 10)) == DONE_EMPTY


def test_solve_accepts_float_point(asset_dir):
    gen = ProblemGenerator(asset_dir)
    problem_id, _, _ = gen.create(seed=1)
    x, y = _inside_point(_answer_of(gen, problem_id).masks[0])
    assert gen.solve(problem_id, (x + 0.3, y - 0.4))["correct"] is True


@pytest.mark.parametrize("dx, correct", [(-HIT_BOX_PX // 2, True), (0, True), (2, True), (HIT_BOX_PX // 2, False)])
def test_hit_box_ratio(dx, correct):
    # 마스크 경계(x=100)에서 dx만큼 떨어진 점: 판정 상자가 마스크에 HIT_RATIO 이상 걸치면 정답
    mask = np.zeros((200, 200), np.uint8)
    mask[:, :100] = 255
    assert _hit(mask, (99 + dx, 100)) is correct


def test_hit_outside_image_is_wrong():
    mask = np.full((200, 200), 255, np.uint8)
    for point in [(-1 - HIT_BOX_PX, 50), (250, 50), (50, -30)]:
        assert _hit(mask, point) is False


def test_without_validator_no_history(asset_dir):
    gen = ProblemGenerator(asset_dir)
    gen.create(seed=1)
    assert gen.history == []
    assert gen.auto_solve_rate() is None


def test_validator_receives_image_and_hints(asset_dir):
    seen = []

    def spy(image, hints):
        seen.append((image, hints))
        return []

    _, image, hints = ProblemGenerator(asset_dir, validator=spy).create(seed=1)
    [(v_image, v_hints)] = seen
    assert np.array_equal(v_image, image)
    assert all(np.array_equal(a, b) for a, b in zip(v_hints, hints))


def test_always_fail_validator(asset_dir):
    gen = ProblemGenerator(asset_dir, validator=always_fail)
    problem_id, _, _ = gen.create(seed=1)
    [record] = gen.history
    assert record["problem_id"] == problem_id and record["seed"] == 1
    assert record["attempts"] == [{"boxes": [], "found": 0, "solved": False}]
    assert record["cap_reached"] is False
    assert gen.auto_solve_rate() == 0.0


def test_always_solve_hits_retry_cap(asset_dir):
    gen = ProblemGenerator(asset_dir, validator=always_solve, max_attempts=3)
    problem_id, image, _ = gen.create(seed=1)
    [record] = gen.history
    assert len(record["attempts"]) == 3
    assert all(a["solved"] and a["found"] == NUM_OBJECTS for a in record["attempts"])
    assert record["cap_reached"] is True
    assert gen.auto_solve_rate() == 1.0
    # 상한에 도달해도 문제는 발급되고, 검수 박스 중심으로 짚으면 모두 맞힌다
    for x1, y1, x2, y2 in always_solve(image, None):
        result = gen.solve(problem_id, ((x1 + x2) / 2, (y1 + y2) / 2))
        assert result["correct"] is True
    assert result["done"] is True and result["found"] == NUM_OBJECTS


def test_partial_find_is_not_solved(asset_dir):
    def finds_two(image, hints):
        return always_solve(image, hints)[:2]

    gen = ProblemGenerator(asset_dir, validator=finds_two)
    gen.create(seed=1)
    [record] = gen.history
    assert record["attempts"][0]["found"] == 2
    assert record["attempts"][0]["solved"] is False and len(record["attempts"]) == 1


def test_regenerates_until_validator_fails(asset_dir):
    calls = []

    def solves_twice(image, hints):
        calls.append(image)
        return always_solve(image, hints) if len(calls) <= 2 else []

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
