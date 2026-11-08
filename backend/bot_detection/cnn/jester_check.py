"""CNN 0단계: Jester(세로 100px)에서 MediaPipe가 손을 찾는지 확인한다. Kaggle 노트북용, 레포 코드에 의존하지 않음.

확인하는 것 (확대 배율 1·2·4배 비교)
- 프레임별 손 검출률
- 연속 10장 이상 손이 잡히는 클립 비율  ← 샘플 1개 = 연속 10장이라 이게 핵심
- 손 크기 (손목~중지 뿌리 거리, 원본 px) → crop 크기를 정할 기준
- 10장 동안 검지 끝 이동량 (손 크기 대비) → 움직임이 충분한 클립이 있는지

Kaggle 사용법
  1. 노트북 오른쪽 Add Input → "20bn-jester" 추가, Settings → Internet ON
  2. !pip install -q mediapipe
  3. 이 파일을 업로드(또는 셀에 붙여넣기) 후
     !python jester_check.py --clips 200
결과: 콘솔 요약 + /kaggle/working/jester_check/ 에 clips.csv, 시각화 PNG
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")          # MediaPipe 경고(W0000 ...) 숨기기
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
WRIST, MIDDLE_MCP, INDEX_TIP = 0, 9, 8
SEQ_LEN = 10            # 샘플 1개 = 연속 10장
CROP_K = 2.0            # crop 한 변 = 손 크기 × CROP_K (1단계 예정값)
GO_RATIO = 0.5          # 연속 10장 잡히는 클립이 이 비율 이상이면 GO


# ---------------------------------------------------------------- 데이터 찾기

def find_clips(root: Path, scan_max: int) -> list[Path]:
    """JPG가 들어 있는 폴더 = 클립 하나.
    빠른 경로: 하위 폴더가 1000개 이상인 폴더(Train 등)를 찾으면 그 목록만 읽는다 (클립 폴더를 열지 않음).
    못 찾으면 전체를 훑되 클립 폴더 안으로는 내려가지 않는다."""
    level = [root]
    for _ in range(5):
        found, nxt = [], []
        for d in level:
            try:
                subs = [Path(e.path) for e in os.scandir(d) if e.is_dir()]
            except OSError:
                continue
            (found if len(subs) >= 1000 else nxt).append((d, subs))
        if found:
            d, subs = max(found, key=lambda x: (x[0].name.lower() == "train", len(x[1])))
            print(f"[데이터] {d} 아래 클립 폴더 목록 사용")
            return sorted(subs)
        level = [s for _, subs in nxt for s in subs]
    clips = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(f.lower().endswith(".jpg") for f in filenames):
            clips.append(Path(dirpath))
            dirnames[:] = []
            if len(clips) >= scan_max:
                break
        dirnames.sort()
    return clips


def load_labels(root: Path) -> dict[str, str]:
    """루트와 한 단계 아래의 CSV에서 {클립 id: 동작 이름}. 형식이 달라 못 읽으면 빈 dict."""
    labels: dict[str, str] = {}
    files = list(root.glob("*.csv")) + list(root.glob("*/*.csv"))
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            if not text:
                continue
            sep = ";" if text[0].count(";") > text[0].count(",") else ","
            rows = list(csv.reader(text, delimiter=sep))
            head = [h.strip().lower() for h in rows[0]]
            if "label" in head:     # 헤더 있음 (예: video_id,label,label_id)
                li = head.index("label")
                ii = next((i for i, h in enumerate(head) if "id" in h and i != li), 0)
                body = rows[1:]
            else:                   # 원본 형식: id;label (헤더 없음)
                ii, li, body = 0, 1, rows
            for r in body:
                if len(r) > max(ii, li):
                    labels[r[ii].strip()] = r[li].strip()
        except Exception as exc:  # noqa: BLE001 — 라벨은 참고용이라 실패해도 진행
            print(f"[라벨] {f.name} 못 읽음: {exc}")
    return labels


def read_frames(clip: Path) -> list[np.ndarray]:
    files = sorted(p for p in clip.iterdir() if p.suffix.lower() == ".jpg")
    frames = [cv2.imread(str(p)) for p in files]
    return [f for f in frames if f is not None]


# ---------------------------------------------------------------- MediaPipe

def get_model(path: Path) -> Path:
    if not path.exists():
        print(f"[모델] 다운로드 → {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def make_landmarker(model: Path, conf: float):
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,   # 프레임마다 새로 검출 = 가장 엄격한 조건
        num_hands=1,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=conf,
    )
    return vision.HandLandmarker.create_from_options(opts)


def detect(landmarker, frame: np.ndarray, scale: int) -> np.ndarray | None:
    """손 랜드마크 21개를 원본 프레임 px 좌표로 (21, 2). 못 찾으면 None."""
    h, w = frame.shape[:2]
    img = frame if scale == 1 else cv2.resize(frame, (w * scale, h * scale),
                                              interpolation=cv2.INTER_CUBIC)
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None
    return np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]], dtype=np.float64)


# ---------------------------------------------------------------- 클립 통계

def longest_run(ok: np.ndarray) -> tuple[int, int]:
    """연속으로 True인 가장 긴 구간 (시작, 길이)."""
    best_s, best_n, s = 0, 0, None
    for i, v in enumerate(list(ok) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s > best_n:
                best_s, best_n = s, i - s
            s = None
    return best_s, best_n


def clip_stats(lms: list[np.ndarray | None]) -> dict:
    ok = np.array([lm is not None for lm in lms])
    s, n = longest_run(ok)
    sizes = [np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]) for lm in lms if lm is not None]
    hand = float(np.median(sizes)) if sizes else float("nan")
    # 가장 긴 연속 구간 안에서, 10장 창마다 검지 끝 이동(시작→끝 직선거리)의 최댓값
    move = float("nan")
    if n >= SEQ_LEN:
        tips = np.array([lms[i][INDEX_TIP] for i in range(s, s + n)])
        move = float(max(np.linalg.norm(tips[j + SEQ_LEN - 1] - tips[j])
                         for j in range(n - SEQ_LEN + 1)))
    return dict(det_rate=float(ok.mean()) if len(ok) else 0.0, run=n, run_start=s,
                has_seq=n >= SEQ_LEN, hand_px=hand, move10_px=move,
                move10_rel=move / hand if hand and not np.isnan(move) else float("nan"))


# ---------------------------------------------------------------- 시각화

def crop_at(frame: np.ndarray, center, side: int) -> np.ndarray:
    """center 중심 side×side crop, 밖은 검정."""
    side = max(4, int(round(side)))
    pad = side
    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    cx, cy = int(round(center[0])) + pad - side // 2, int(round(center[1])) + pad - side // 2
    return padded[cy:cy + side, cx:cx + side]


def save_preview(path: Path, frames, lms, st: dict, title: str) -> None:
    """윗줄: 원본 프레임에 랜드마크·crop 상자 (4배 확대). 아랫줄: 1단계 예정 crop 10장 (64×64 흑백)."""
    s, n = st["run_start"], st["run"]
    idx = list(range(s, s + min(n, SEQ_LEN))) if n else list(range(min(len(frames), SEQ_LEN)))
    top, bottom = [], []
    for i in idx:
        f = cv2.resize(frames[i], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        lm = lms[i]
        if lm is not None:
            for x, y in lm * 4:
                cv2.circle(f, (int(x), int(y)), 3, (0, 255, 0), -1)
            side = np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]) * CROP_K
            tx, ty = lm[INDEX_TIP]
            cv2.rectangle(f, (int((tx - side / 2) * 4), int((ty - side / 2) * 4)),
                          (int((tx + side / 2) * 4), int((ty + side / 2) * 4)), (0, 0, 255), 2)
            c = crop_at(frames[i], (tx, ty), side)
            c = cv2.cvtColor(cv2.resize(c, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            bottom.append(cv2.cvtColor(c, cv2.COLOR_GRAY2BGR))
        else:
            bottom.append(np.zeros((64, 64, 3), np.uint8))
        top.append(cv2.resize(f, (240, int(240 * f.shape[0] / f.shape[1]))))
    th = max(t.shape[0] for t in top)
    top = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in top]
    bottom = [cv2.copyMakeBorder(b, 0, 0, 0, 244 - 64, cv2.BORDER_CONSTANT) for b in bottom]
    sheet = np.vstack([np.hstack(top), np.hstack(bottom)])
    sheet = cv2.copyMakeBorder(sheet, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(sheet, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet)


# ---------------------------------------------------------------- 메인

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input", help="Jester 데이터 위치 (하위 폴더 자동 탐색)")
    ap.add_argument("--out", default=None, help="기본: /kaggle/working/jester_check_<클립수>c_<배율> (실행마다 안 덮어쓰게)")
    ap.add_argument("--clips", type=int, default=200, help="검사할 클립 수")
    ap.add_argument("--scan-max", type=int, default=20000, help="이만큼 클립 폴더를 찾은 뒤 그 안에서 랜덤 추출")
    ap.add_argument("--scales", default="1,2,4")
    ap.add_argument("--conf", type=float, default=0.5, help="MediaPipe 검출 신뢰도 기준")
    ap.add_argument("--previews", type=int, default=8)
    ap.add_argument("--model", default="/kaggle/working/hand_landmarker.task")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scales = [int(s) for s in args.scales.split(",")]
    root = Path(args.root)
    out = Path(args.out or f"/kaggle/working/jester_check_{args.clips}c_{'-'.join(map(str, scales))}x")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    clips = find_clips(root, args.scan_max)
    print(f"[데이터] 클립 폴더 {len(clips)}개 발견 ({time.time() - t0:.0f}s)")
    if not clips:
        raise SystemExit(f"{root} 아래에 JPG 폴더가 없음. --root 확인")
    labels = load_labels(root) or load_labels(clips[0].parents[1])
    print(f"[라벨] {len(labels)}개 읽음" + ("" if labels else " (없음, 동작별 통계 생략)"))
    random.seed(args.seed)
    clips = random.sample(clips, min(args.clips, len(clips)))
    sample = read_frames(clips[0])
    print(f"[데이터] 예시 경로: {clips[0]}, 프레임 크기: {sample[0].shape[1::-1] if sample else '?'}")

    model = get_model(Path(args.model))
    landmarkers = {s: make_landmarker(model, args.conf) for s in scales}
    rows, cache, cost = [], {}, {s: [] for s in scales}

    for k, clip in enumerate(clips):
        frames = read_frames(clip)
        if not frames:
            continue
        for s in scales:
            t = time.time()
            lms = [detect(landmarkers[s], f, s) for f in frames]
            cost[s].append((time.time() - t) / len(frames))
            st = clip_stats(lms)
            rows.append(dict(clip=clip.name, label=labels.get(clip.name, "?"), scale=s,
                             frames=len(frames), **st))
            cache[(clip, s)] = lms
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(clips)} 클립 ({time.time() - t0:.0f}s)")

    with open(out / "clips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 배율별 요약
    print("\n배율 | 프레임 검출률 | 연속10장 클립 | 손 크기 px(중앙) | 10장 이동/손크기(중앙) | ms/프레임")
    best, best_ratio = scales[0], -1.0
    for s in scales:
        r = [x for x in rows if x["scale"] == s]
        seq = [x for x in r if x["has_seq"]]
        ratio = len(seq) / len(r)
        hand = np.nanmedian([x["hand_px"] for x in r]) if any(not np.isnan(x["hand_px"]) for x in r) else float("nan")
        mv = np.nanmedian([x["move10_rel"] for x in seq]) if seq else float("nan")
        print(f" {s}x  | {np.mean([x['det_rate'] for x in r]):>11.1%} | {ratio:>11.1%} | "
              f"{hand:>14.1f} | {mv:>20.2f} | {1000 * np.mean(cost[s]):>7.1f}")
        if ratio > best_ratio:
            best, best_ratio = s, ratio

    # ---- 동작별 (최적 배율)
    if labels:
        print(f"\n동작별 연속10장 비율 ({best}x, 상위·하위)")
        by: dict[str, list] = {}
        for x in rows:
            if x["scale"] == best:
                by.setdefault(x["label"], []).append(x)
        stat = sorted(((float(np.mean([x["has_seq"] for x in v])), len(v), lab)
                       for lab, v in by.items()), reverse=True)
        for m, n, lab in stat:
            print(f"  {m:>5.0%}  n={n:<3} {lab}")

    # ---- 시각화: 최적 배율에서 연속 10장 잡힌 클립 우선
    ok = [x for x in rows if x["scale"] == best and x["has_seq"]]
    bad = [x for x in rows if x["scale"] == best and not x["has_seq"]]
    n_bad = min(2, len(bad))
    pick = ok[:args.previews - n_bad] + bad[:n_bad]
    for x in pick:
        clip = next(c for c in clips if c.name == x["clip"])
        save_preview(out / f"{'ok' if x['has_seq'] else 'fail'}_{x['clip']}.png", read_frames(clip),
                     cache[(clip, best)], dict(x),
                     f"{x['clip']} {x['label']} {best}x det={x['det_rate']:.0%} run={x['run']}")
    print(f"\n[저장] {out}/clips.csv, 미리보기 PNG {len(pick)}장")

    verdict = "GO" if best_ratio >= GO_RATIO else "NO-GO (계획 재검토)"
    print(f"\n판정: {verdict} — 최적 {best}x에서 연속 10장 잡히는 클립 {best_ratio:.1%} (기준 {GO_RATIO:.0%})")
    print(f"총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
