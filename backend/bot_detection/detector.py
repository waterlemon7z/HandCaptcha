"""진위검출 최종 판정.

손인식이 done을 내면 그 HandResult를 check()에 넘긴다. 결과는 세 가지:
- True  = 사람
- False = 봇 (검사 하나라도 걸림)
- None  = 재측정 (손 변형 검사를 할 데이터가 없음 → "손을 조금 더 크게 움직여 다시 해주세요")
원칙은 보수적(사람을 봇으로 오판하지 않기): 각 검사는 확실한 봇만 거른다.
단, 핵심 검사(②' 손 변형)를 못 하면 통과가 아니라 재측정. 짧게 움직이거나 프레임을 끊어 검사를 피하는 봇 대응.
재측정이 MAX_RETRY번 연속이면 호출하는 쪽(worker)에서 실패 처리 (통과로 하면 봇이 끝까지 짧게 움직여 뚫음).

1단계 궤적 (trajectory.py)
- jitter_rms < 0.4                                → 봇 "멈출 때 떨림이 거의 없음"
- straightness > 0.95 and jitter_lag1 < -0.3      → 봇 "직선 경로 + 매 프레임 따로 튀는 떨림"
2단계 영상 (frame_check.py)
- crop이 없음                                      → 봇 "검증할 영상이 없음"
- affine_res < 0.08                               → 봇 "손이 사진처럼 통째로만 움직임 (변형 없음)"
- (① 배경 노이즈 zero_ratio는 2026-11-08 판정에서 뺌, 참고값으로만 reason에 표시.
   노이즈 제거·압축을 하는 노트북 웹캠에서 진짜 사람이 0.72~0.78로 봇 판정됨.
   우리 데이터에서 ①로만 잡히던 봇은 0개 — ②'가 전부 잡음)
- affine_res 측정 불가 (쓸 이동 쌍 5개 미만)          → 재측정 (단, CNN이 봇이라 하면 봇)
  (쌍에서 뺌: 0.15초 넘게 벌어진 쌍, 픽셀까지 같은 중복 프레임, 크기 깨진 crop)
3단계 CNN (cnn/runtime.py, model.onnx를 cv2.dnn으로 — PyTorch 불필요)
- 가짜 확률 > 0.976                                → 봇 "CNN: 손이 사진처럼 움직임"
- 10장 묶음이 안 나오거나 model.onnx가 없으면          → CNN만 생략

사용
    from backend.bot_detection.detector import check, MAX_RETRY
    is_human, reason = check(result)      # result: done 상태의 HandResult
    if is_human is None: ...              # 재측정 안내 후 다시 기록 (연속 MAX_RETRY번이면 실패 처리)
"""
from __future__ import annotations

import math

import numpy as np

from backend.bot_detection import frame_check, trajectory

# 기준값 (2026-11-08, 사람 89개 = 640×480 25개 + 1280×720 64개, 합성 봇 1280×720 64개)
JITTER_MIN = 0.4        # 멈춤 구간 떨림(px)이 이보다 작으면 봇. 사람 최소 0.48
STRAIGHT_MAX = 0.95     # 경로 곧음이 이보다 크고
LAG1_MIN = -0.3         #   떨림 자기상관이 이보다 작으면 봇. 사람 최댓값 쪽 (0.91, -0.27) 하나 있어서 좁힘
AFFINE_MIN = 0.08       # 변환 맞춤 오차(px)가 이보다 작으면 봇. 사람 최소 0.33 (쌍 5개 이상일 때), 합성 봇 최대 0.06
AFFINE_MIN_PAIRS = 5    # ②'에 쓴 이동 쌍이 이보다 적으면 측정 불가로 봄 (2쌍으로 봇 판정된 사람 있었음)
MAX_RETRY = 2           # 재측정이 이만큼 연속이면 실패 처리 (호출하는 쪽에서)
USE_CNN = True          # cnn/model.onnx가 있으면 CNN 검사도 함 (없으면 자동 생략)
CNN_THRESHOLD = 0.976   # CNN 가짜 확률이 이보다 크면 봇. Jester 검증에서 사람 오판 ≤1%가 되는 문턱

RETRY_MESSAGE = "손을 조금 더 크게 움직여 다시 해주세요"


def check(result) -> tuple[bool | None, str]:
    """done 상태의 HandResult를 검사한다. (True 사람 / False 봇 / None 재측정, 이유)."""
    return check_arrays(result.raw_trajectory, result.timestamps, result.frames, result.frame_size,
                        getattr(result, "frame_timestamps", None))


def cnn_score(frames, frame_times, raw_trajectory, timestamps) -> tuple[float | None, int]:
    """CNN 가짜 확률 (묶음 중앙값, 묶음 수). model.onnx가 없거나 묶음이 없으면 (None, 0)."""
    from backend.bot_detection.cnn.runtime import get_scorer
    scorer = get_scorer() if USE_CNN else None
    if scorer is None:
        return None, 0
    return scorer.score(frames, frame_times, raw_trajectory, timestamps)


def check_arrays(raw_trajectory, timestamps, frames, frame_size, frame_times=None) -> tuple[bool | None, str]:
    """check()의 배열 버전. 저장된 녹화(npz)를 검사할 때 쓴다."""
    raw = np.asarray(raw_trajectory, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if len(raw) < 3:
        return False, "궤적이 너무 짧음"

    t = trajectory.extract(raw, times)
    if t.jitter_rms < JITTER_MIN:
        return False, f"멈출 때 떨림이 거의 없음 (jitter_rms {t.jitter_rms:.2f}px < {JITTER_MIN})"
    if t.straightness > STRAIGHT_MAX and t.jitter_lag1 < LAG1_MIN:
        return False, (f"직선 경로 + 매 프레임 따로 튀는 떨림 "
                       f"(straightness {t.straightness:.2f}, jitter_lag1 {t.jitter_lag1:.2f})")

    if frames is None or len(frames) == 0 or frame_size is None or tuple(frame_size) == (0, 0):
        return False, "검증할 영상이 없음"

    f = frame_check.extract(frames, raw, times, tuple(frame_size))
    affine_ok = not math.isnan(f.affine_res) and f.deform_pairs >= AFFINE_MIN_PAIRS
    if affine_ok and f.affine_res < AFFINE_MIN:
        return False, f"손이 사진처럼 통째로만 움직임 (affine_res {f.affine_res:.3f}px < {AFFINE_MIN})"

    cnn, windows_n = cnn_score(frames, frame_times, raw, times)
    if cnn is not None and cnn > CNN_THRESHOLD:
        return False, f"CNN: 손이 사진처럼 움직임 (가짜 확률 {cnn:.3f} > {CNN_THRESHOLD}, 묶음 {windows_n}개)"

    if not affine_ok:
        return None, (f"재측정: {RETRY_MESSAGE} "
                      f"(손 변형 검사에 쓸 이동 쌍 {f.deform_pairs}개 < {AFFINE_MIN_PAIRS})")

    notes = [f"affine_res {f.affine_res:.3f}",
             f"CNN {cnn:.3f}" if cnn is not None else "CNN 생략(10장 묶음 없음)"]
    if not math.isnan(f.zero_ratio):
        notes.append(f"배경 노이즈 {f.zero_ratio:.2f} 참고")
    return True, f"사람 ({', '.join(notes)})"


if __name__ == "__main__":
    # 저장된 데이터 전체 판정
    import glob
    from collections import Counter
    from pathlib import Path

    from backend.bot_detection.mock import load_csv
    from backend.bot_detection.record import load_recording

    label = {True: "사람", False: "봇", None: "재측정"}
    data_dir = Path(__file__).resolve().parent / "test" / "data"
    results: dict[str, list[tuple[str, bool | None, str]]] = {}
    for f in sorted(glob.glob(str(data_dir / "bot" / "*.csv"))):
        m = load_csv(f)
        results.setdefault("좌표봇", []).append((Path(f).stem, *check_arrays(m.raw_trajectory, m.timestamps, None, None)))
    for f in sorted(glob.glob(str(data_dir / "human" / "*.npz"))) + \
            sorted(glob.glob(str(data_dir / "synth" / "S*" / "*.npz"))):
        d = load_recording(f)
        ok = check_arrays(d["raw_trajectory"], d["timestamps"], d["frames"], d["frame_size"],
                          d.get("frame_timestamps"))
        results.setdefault(Path(f).parent.name, []).append((Path(f).stem, *ok))

    for g, rs in results.items():
        cnt = Counter(ok for _, ok, _ in rs)
        print(f"[{g}] {len(rs)}개 중 사람 {cnt[True]}, 봇 {cnt[False]}, 재측정 {cnt[None]}")
        reasons = Counter(r.split(" (")[0] for _, ok, r in rs if ok is False)
        for r, n in reasons.most_common():
            print(f"    봇 이유: {r} × {n}")
        for name, ok, r in rs:
            if (g == "human" and ok is not True) or (g != "human" and ok is not False):
                print(f"    {name}: {label[ok]} — {r}")
