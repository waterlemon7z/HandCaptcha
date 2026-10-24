import cv2
import numpy as np

from cv.problem_gen import ProblemGenerator
from cv.problem_gen.generator import OUTPUT_SIZE

from conftest import always_fail, always_solve

W, H = OUTPUT_SIZE


def _answer_of(gen, problem_id):
    """테스트용: 내부에 저장된 정답 (외부 API로는 판정 전에 볼 수 없음)."""
    return gen._answers[problem_id]


def _inside_point(mask):
    # 마스크를 깎아 낸 뒤의 점을 골라 경계 반올림 영향을 피한다
    ys, xs = np.nonzero(cv2.erode(mask, np.ones((5, 5), np.uint8)))
    return int(xs[len(xs) // 2]), int(ys[len(ys) // 2])


def test_create_returns_only_id_and_images(asset_dir):
    out = ProblemGenerator(asset_dir).create(seed=1)
    assert len(out) == 3
    problem_id, left, right = out
    assert isinstance(problem_id, str)
    assert left.shape == right.shape == (H, W, 3)
    assert left.dtype == right.dtype == np.uint8


def test_same_seed_same_problem_new_id(asset_dir):
    gen = ProblemGenerator(asset_dir)
    id1, l1, r1 = gen.create(seed=7)
    id2, l2, r2 = gen.create(seed=7)
    assert id1 != id2
    assert np.array_equal(l1, l2) and np.array_equal(r1, r2)
    assert np.array_equal(_answer_of(gen, id1).mask, _answer_of(gen, id2).mask)

    _, _, r3 = gen.create(seed=8)
    assert not np.array_equal(r1, r3)


def test_problem_ids_are_unique(asset_dir):
    gen = ProblemGenerator(asset_dir)
    assert len({gen.create(seed=0)[0] for _ in range(20)}) == 20


def test_solve_correct_reveals_answer(asset_dir):
    gen = ProblemGenerator(asset_dir)
    problem_id, _, _ = gen.create(seed=1)
    mask = _answer_of(gen, problem_id).mask
    result = gen.solve(problem_id, _inside_point(mask))
    assert result["correct"] is True
    assert np.array_equal(result["answer_mask"], mask)
    assert result["answer_box"] == cv2.boundingRect(mask)


def test_solve_accepts_float_point(asset_dir):
    gen = ProblemGenerator(asset_dir)
    problem_id, _, _ = gen.create(seed=1)
    x, y = _inside_point(_answer_of(gen, problem_id).mask)
    assert gen.solve(problem_id, (x + 0.3, y - 0.4))["correct"] is True


def test_solve_wrong_still_reveals_answer(asset_dir):
    gen = ProblemGenerator(asset_dir)
    for point in [(0, 0), (-5, 10), (W + 3, H // 2)]:
        problem_id, _, _ = gen.create(seed=1)
        result = gen.solve(problem_id, point)
        assert result["correct"] is False
        assert result["answer_mask"] is not None and result["answer_box"] is not None


def test_hit_tolerance_accepts_near_miss(asset_dir):
    gen = ProblemGenerator(asset_dir)
    problem_id, _, _ = gen.create(seed=1)
    x, y, w, h = _answer_of(gen, problem_id).box
    # 물체가 원이라 박스 왼쪽 변 가운데는 마스크 경계 바로 옆
    assert gen.solve(problem_id, (x - 4, y + h // 2))["correct"] is True


def test_solved_id_is_discarded(asset_dir):
    gen = ProblemGenerator(asset_dir)
    problem_id, _, _ = gen.create(seed=1)
    point = _inside_point(_answer_of(gen, problem_id).mask)
    assert gen.solve(problem_id, point)["correct"] is True
    assert gen.solve(problem_id, point) == {"correct": False, "answer_mask": None, "answer_box": None}
    assert gen.solve("unknown-id", point) == {"correct": False, "answer_mask": None, "answer_box": None}


def test_object_stays_inside_frame(asset_dir):
    # 오른쪽 이미지를 확대·회전해도 물체가 잘리지 않아야 한다
    gen = ProblemGenerator(asset_dir)
    for seed in range(60):
        problem_id, _, _ = gen.create(seed=seed)
        answer = _answer_of(gen, problem_id)
        x, y, w, h = answer.box
        assert x > 0 and y > 0 and x + w < W and y + h < H, seed


def test_without_validator_no_history(asset_dir):
    gen = ProblemGenerator(asset_dir)
    gen.create(seed=1)
    assert gen.history == []
    assert gen.auto_solve_rate() is None


def test_always_fail_validator(asset_dir):
    gen = ProblemGenerator(asset_dir, validator=always_fail)
    problem_id, _, _ = gen.create(seed=1)
    [record] = gen.history
    assert record["problem_id"] == problem_id and record["seed"] == 1
    assert record["attempts"] == [{"predicted": None, "solved": False}]
    assert record["cap_reached"] is False
    assert gen.auto_solve_rate() == 0.0


def test_always_solve_hits_retry_cap(asset_dir):
    gen = ProblemGenerator(asset_dir, validator=always_solve, max_attempts=3)
    problem_id, _, right = gen.create(seed=1)
    [record] = gen.history
    assert len(record["attempts"]) == 3
    assert all(a["solved"] for a in record["attempts"])
    assert record["cap_reached"] is True
    assert gen.auto_solve_rate() == 1.0
    # 상한에 도달해도 문제는 발급되고 판정할 수 있다
    assert gen.solve(problem_id, always_solve(None, right))["correct"] is True


def test_regenerates_until_validator_fails(asset_dir):
    calls = []

    def solves_twice(left, right):
        calls.append(right)
        return always_solve(left, right) if len(calls) <= 2 else None

    gen = ProblemGenerator(asset_dir, validator=solves_twice)
    _, _, right = gen.create(seed=1)
    [record] = gen.history
    assert [a["solved"] for a in record["attempts"]] == [True, True, False]
    assert record["cap_reached"] is False
    assert np.array_equal(right, calls[-1])
    assert not np.array_equal(calls[0], calls[-1])
    assert gen.auto_solve_rate() == 1.0  # 첫 시도에 풀렸으므로


def test_same_seed_same_result_with_validator(asset_dir):
    gen = ProblemGenerator(asset_dir, validator=always_solve, max_attempts=3)
    _, _, r1 = gen.create(seed=5)
    _, _, r2 = gen.create(seed=5)
    assert np.array_equal(r1, r2)
