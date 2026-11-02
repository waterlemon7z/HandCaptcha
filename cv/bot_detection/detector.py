"""진위검출 최종 판정.

손인식이 done을 내면 그 HandResult를 check()에 넘긴다. 하나라도 걸리면 봇.
원칙은 보수적(사람을 봇으로 오판하지 않기): 각 검사는 확실한 봇만 거르고, 잴 수 없으면 통과시킨다.

1단계 궤적 (trajectory.py)
- jitter_rms < 0.5                                → "멈출 때 떨림이 거의 없음"
- straightness > 0.9 and jitter_lag1 < -0.2       → "직선 경로 + 매 프레임 따로 튀는 떨림"
2단계 영상 (frame_check.py)
- crop이 없음                                      → "검증할 영상이 없음"
- zero_ratio > 0.65                               → "배경이 픽셀 단위로 정지 (실제 카메라 아님)"
- affine_res < 0.08                               → "손이 사진처럼 통째로만 움직임 (변형 없음)"

사용
    from cv.bot_detection.detector import check
    is_human, reason = check(result)      # result: done 상태의 HandResult
"""
from __future__ import annotations

import math

import numpy as np

from cv.bot_detection import frame_check, trajectory

JITTER_MIN = 0.5        # 멈춤 구간 떨림(px)이 이보다 작으면 봇. 사람 최소 0.70
STRAIGHT_MAX = 0.9      # 경로 곧음이 이보다 크고
LAG1_MIN = -0.2         #   떨림 자기상관이 이보다 작으면 봇. 사람 중 둘 다 해당 없음
ZERO_MAX = 0.65         # 배경 차이 0 비율이 이보다 크면 봇. 사람 최대 0.50
AFFINE_MIN = 0.08       # 변환 맞춤 오차(px)가 이보다 작으면 봇. 사람 최소 0.17


def check(result) -> tuple[bool, str]:
    """done 상태의 HandResult를 검사한다. (사람이면 True, 이유)."""
    return check_arrays(result.raw_trajectory, result.timestamps, result.frames, result.frame_size)


def check_arrays(raw_trajectory, timestamps, frames, frame_size) -> tuple[bool, str]:
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
    if not math.isnan(f.zero_ratio) and f.zero_ratio > ZERO_MAX:
        return False, f"배경이 픽셀 단위로 정지, 실제 카메라 아님 (zero_ratio {f.zero_ratio:.2f} > {ZERO_MAX})"
    if not math.isnan(f.affine_res) and f.affine_res < AFFINE_MIN:
        return False, f"손이 사진처럼 통째로만 움직임 (affine_res {f.affine_res:.3f}px < {AFFINE_MIN})"

    skipped = [name for name, v in (("배경 노이즈", f.zero_ratio), ("손 변형", f.affine_res))
               if math.isnan(v)]
    return True, "사람" + (f" ({', '.join(skipped)} 검사는 데이터 부족으로 생략)" if skipped else "")


if __name__ == "__main__":
    # 저장된 데이터 전체 판정
    import glob
    from collections import Counter
    from pathlib import Path

    from cv.bot_detection.mock import load_csv
    from cv.bot_detection.record import load_recording

    data_dir = Path(__file__).resolve().parents[1] / "test" / "data"
    results: dict[str, list[tuple[str, bool, str]]] = {}
    for f in sorted(glob.glob(str(data_dir / "bot" / "*.csv"))):
        m = load_csv(f)
        results.setdefault("좌표봇", []).append((Path(f).stem, *check_arrays(m.raw_trajectory, m.timestamps, None, None)))
    for f in sorted(glob.glob(str(data_dir / "human" / "*.npz"))) + \
            sorted(glob.glob(str(data_dir / "synth" / "S*" / "*.npz"))):
        d = load_recording(f)
        ok = check_arrays(d["raw_trajectory"], d["timestamps"], d["frames"], d["frame_size"])
        results.setdefault(Path(f).parent.name, []).append((Path(f).stem, *ok))

    for g, rs in results.items():
        humans = sum(ok for _, ok, _ in rs)
        print(f"[{g}] {len(rs)}개 중 사람 판정 {humans}, 봇 판정 {len(rs) - humans}")
        reasons = Counter(r.split(" (")[0] for _, ok, r in rs if not ok)
        for r, n in reasons.most_common():
            print(f"    봇 이유: {r} × {n}")
        if g != "human":
            for name, ok, r in rs:
                if ok:
                    print(f"    통과: {name} — {r}")
        else:
            for name, ok, r in rs:
                if not ok or "생략" in r:
                    print(f"    {name}: {ok} — {r}")
