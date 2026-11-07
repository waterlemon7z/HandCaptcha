"""CNN 1단계: Jester 클립에서 '진짜 손' 샘플을 만든다. Kaggle 노트북용, 레포 코드에 의존하지 않음.

샘플 1개 = 손이 움직이는 연속 10장, 프레임마다 검지 끝 중심 crop → 64×64 흑백 → (10, 64, 64) uint8
- crop 한 변 = 손 크기(손목~중지 뿌리) × K, K는 샘플마다 1.2~2.5 랜덤 (우리 앱 crop 크기가 아직 미정이라 넓게)
- 10장 안에서 crop 크기는 고정, 창은 검지 끝을 따라감 (우리 앱 HandTracker crop과 같은 방식)
- 2단계(가짜 만들기)용으로 클립 id, 시작 프레임, 랜드마크 10장, crop 크기도 같이 저장
- 학습/검증 분리는 클립 id로 (10개 중 1개가 검증)

Kaggle 사용법
  1. Add Input → "20bn-jester", Settings → Internet ON
  2. 셀 1: %%writefile jester_prep.py + 이 파일 전체
  3. 셀 2: !pip install -q mediapipe
           !python -u jester_prep.py --clips 8000
  4. 오래 걸리니 오른쪽 위 Save Version → "Save & Run All (Commit)"으로 돌리면
     창을 닫아도 돌아가고, 결과(/kaggle/working)가 버전에 남아 2단계 노트북에서 Input으로 붙일 수 있음
결과: /kaggle/working/jester_real/real_*.npz, summary.csv, preview.png
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
import urllib.request
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")          # MediaPipe 경고 숨기기
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
WRIST, MIDDLE_MCP, INDEX_TIP = 0, 9, 8
SEQ_LEN = 10            # 샘플 1개 = 연속 10장
SIZE = 64               # crop을 이 크기로 줄임
STRIDE = 5              # 연속 구간 안에서 샘플 시작점 간격 (후보)


# ---------------------------------------------------------------- 데이터 찾기 (0단계와 같음)

def find_clips(root: Path) -> list[Path]:
    """하위 폴더가 1000개 이상인 폴더(Train 등)의 목록. 클립 폴더를 열지 않아 빠름."""
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
    return []


def load_labels(root: Path) -> dict[str, str]:
    """루트와 한 단계 아래의 CSV에서 {클립 id: 동작 이름}."""
    labels: dict[str, str] = {}
    for f in list(root.glob("*.csv")) + list(root.glob("*/*.csv")):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            if not text:
                continue
            sep = ";" if text[0].count(";") > text[0].count(",") else ","
            rows = list(csv.reader(text, delimiter=sep))
            head = [h.strip().lower() for h in rows[0]]
            if "label" in head:
                li = head.index("label")
                ii = next((i for i, h in enumerate(head) if "id" in h and i != li), 0)
                body = rows[1:]
            else:
                ii, li, body = 0, 1, rows
            for r in body:
                if len(r) > max(ii, li):
                    labels[r[ii].strip()] = r[li].strip()
        except Exception as exc:  # noqa: BLE001
            print(f"[라벨] {f.name} 못 읽음: {exc}")
    return labels


def read_frames(clip: Path) -> list[np.ndarray]:
    files = sorted(p for p in clip.iterdir() if p.suffix.lower() == ".jpg")
    frames = [cv2.imread(str(p)) for p in files]
    return [f for f in frames if f is not None]


def get_model(path: Path) -> Path:
    if not path.exists():
        print(f"[모델] 다운로드 → {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


# ---------------------------------------------------------------- 작업 프로세스 (코어마다 1개)

_LM = None
_CFG: dict = {}


def init_worker(model: str, conf: float, cfg: dict) -> None:
    global _LM, _CFG
    import mediapipe as mp  # noqa: F401 — 프로세스마다 따로 불러옴
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=conf,
    )
    _LM = vision.HandLandmarker.create_from_options(opts)
    _CFG = cfg


def detect(frame: np.ndarray) -> np.ndarray | None:
    """손 랜드마크 21개 (원본 px). 못 찾으면 None."""
    import mediapipe as mp
    h, w = frame.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = _LM.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None
    return np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]], dtype=np.float32)


def runs(ok: np.ndarray) -> list[tuple[int, int]]:
    """연속으로 True인 구간들 (시작, 길이)."""
    out, s = [], None
    for i, v in enumerate(list(ok) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - s))
            s = None
    return out


def crop_at(frame: np.ndarray, center, side: float) -> np.ndarray:
    """center 중심 side×side crop, 화면 밖은 검정 (HandTracker crop_around와 같은 방식).
    center가 화면 밖 멀리 있어도 항상 side×side를 돌려준다 (MediaPipe 랜드마크는 화면 밖 좌표가 나올 수 있음)."""
    side = max(4, int(round(side)))
    h, w = frame.shape[:2]
    out = np.zeros((side, side) + frame.shape[2:], frame.dtype)
    x0 = int(round(center[0])) - side // 2
    y0 = int(round(center[1])) - side // 2
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out


def to_sample(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def process_clip(task) -> dict:
    """클립 하나 → 샘플 여러 개. 에러가 나면 그 클립만 건너뜀 (전체 실행은 계속)."""
    try:
        return _process_clip(task)
    except Exception as exc:  # noqa: BLE001
        return dict(clip=Path(task[0]).name, label=task[1], frames=0, detected=0, samples=[],
                    error=f"{type(exc).__name__}: {exc}"[:200])


def _process_clip(task) -> dict:
    clip, label, seed = task
    cfg = _CFG
    rng = np.random.default_rng(seed)
    frames = read_frames(Path(clip))
    n = len(frames)
    info = dict(clip=Path(clip).name, label=label, frames=n, detected=0, samples=[])
    if n < SEQ_LEN:
        return info
    lms = np.full((n, 21, 2), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        lm = detect(f)
        if lm is not None:
            lms[i] = lm
    ok = ~np.isnan(lms[:, 0, 0])
    info["detected"] = int(ok.sum())

    starts = [st for s, m in runs(ok) if m >= SEQ_LEN for st in range(s, s + m - SEQ_LEN + 1, STRIDE)]
    chosen: list[int] = []
    for st in rng.permutation(starts) if starts else []:
        st = int(st)
        if any(abs(st - c) < SEQ_LEN for c in chosen):     # 샘플끼리 프레임 안 겹치게
            continue
        win = lms[st:st + SEQ_LEN]
        hand = float(np.median(np.linalg.norm(win[:, MIDDLE_MCP] - win[:, WRIST], axis=1)))
        if hand < cfg["min_hand"]:
            continue
        tips = win[:, INDEX_TIP]
        extent = float((tips.max(0) - tips.min(0)).max())
        if extent < cfg["min_move"] * hand:                # 손이 거의 안 움직이면 제외 (우리 ②도 이동 구간만 봄)
            continue
        k = float(rng.uniform(cfg["k_min"], cfg["k_max"]))
        side = hand * k
        x = np.stack([to_sample(crop_at(frames[st + j], tips[j], side)) for j in range(SEQ_LEN)])
        info["samples"].append(dict(x=x, start=st, side=side, k=k, hand=hand, move=extent / hand, lms=win))
        chosen.append(st)
        if len(chosen) >= cfg["per_clip"]:
            break
    return info


# ---------------------------------------------------------------- 저장

def split_of(clip_id: str) -> int:
    """0 = 학습, 1 = 검증 (클립 id 기준 10개 중 1개)."""
    return int(zlib.crc32(clip_id.encode()) % 10 == 0)


def save_shard(path: Path, buf: list[dict]) -> None:
    np.savez_compressed(
        path,
        x=np.stack([b["x"] for b in buf]).astype(np.uint8),            # (n, 10, 64, 64)
        clip=np.array([b["clip"] for b in buf]),
        label=np.array([b["label"] for b in buf]),
        start=np.array([b["start"] for b in buf], dtype=np.int32),     # 클립 안 시작 프레임 (0부터)
        side=np.array([b["side"] for b in buf], dtype=np.float32),     # crop 한 변 (원본 px)
        k=np.array([b["k"] for b in buf], dtype=np.float32),
        hand=np.array([b["hand"] for b in buf], dtype=np.float32),     # 손 크기 (원본 px)
        move=np.array([b["move"] for b in buf], dtype=np.float32),     # 10장 동안 검지 이동 범위 / 손 크기
        lms=np.stack([b["lms"] for b in buf]).astype(np.float32),      # (n, 10, 21, 2) 원본 px
        split=np.array([split_of(b["clip"]) for b in buf], dtype=np.uint8),
    )


def save_preview(path: Path, buf: list[dict], rows: int = 8) -> None:
    """샘플마다 한 줄 = 10장 (2배 확대)."""
    lines = []
    for b in buf[:rows]:
        line = np.hstack([cv2.copyMakeBorder(f, 0, 2, 0, 2, cv2.BORDER_CONSTANT, value=0) for f in b["x"]])
        line = cv2.resize(line, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        line = cv2.copyMakeBorder(cv2.cvtColor(line, cv2.COLOR_GRAY2BGR), 22, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(line, f"{b['clip']} {b['label']} k={b['k']:.2f} move={b['move']:.2f}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        lines.append(line)
    if lines:
        cv2.imwrite(str(path), np.vstack(lines))


# ---------------------------------------------------------------- 메인

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/jester_real")
    ap.add_argument("--clips", type=int, default=8000, help="처리할 클립 수")
    ap.add_argument("--offset", type=int, default=0, help="섞은 클립 목록에서 여기부터 (이어서 더 만들 때)")
    ap.add_argument("--per-clip", type=int, default=3, help="클립당 최대 샘플 수")
    ap.add_argument("--k-min", type=float, default=1.2)
    ap.add_argument("--k-max", type=float, default=2.5)
    ap.add_argument("--min-move", type=float, default=0.25, help="10장 동안 검지 이동 범위 / 손 크기 최소")
    ap.add_argument("--min-hand", type=float, default=8.0, help="손 크기(px) 최소. 너무 작은 손 제외")
    ap.add_argument("--exclude", default="No gesture", help="뺄 동작 (쉼표 구분). No gesture는 2단계 배경용")
    ap.add_argument("--shard", type=int, default=2000, help="npz 파일 하나당 샘플 수")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--model", default="/kaggle/working/hand_landmarker.task")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    clips = find_clips(Path(args.root))
    if not clips:
        raise SystemExit(f"{args.root} 아래에서 클립 폴더를 못 찾음")
    labels = load_labels(Path(args.root)) or load_labels(clips[0].parents[1])
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    clips = [c for c in clips if labels.get(c.name, "?") not in exclude]
    random.Random(args.seed).shuffle(clips)
    clips = clips[args.offset:args.offset + args.clips]
    print(f"[데이터] 클립 {len(clips)}개 처리 (제외: {sorted(exclude)}), 코어 {args.workers}개")

    model = str(get_model(Path(args.model)))
    cfg = dict(per_clip=args.per_clip, k_min=args.k_min, k_max=args.k_max,
               min_move=args.min_move, min_hand=args.min_hand)
    tasks = [(str(c), labels.get(c.name, "?"), args.seed * 1_000_003 + args.offset + i)
             for i, c in enumerate(clips)]

    buf, shard_no, total = [], 0, 0
    stats: list[dict] = []
    preview: list[dict] = []
    with Pool(args.workers, initializer=init_worker, initargs=(model, args.conf, cfg)) as pool:
        for k, info in enumerate(pool.imap_unordered(process_clip, tasks, chunksize=4), 1):
            for s in info["samples"]:
                buf.append(dict(s, clip=info["clip"], label=info["label"]))
            stats.append(dict(clip=info["clip"], label=info["label"], frames=info["frames"],
                              detected=info["detected"], samples=len(info["samples"]),
                              error=info.get("error", "")))
            if info.get("error") and sum(1 for s in stats if s["error"]) <= 5:
                print(f"  [건너뜀] {info['clip']}: {info['error']}")
            if len(preview) < 8 and info["samples"]:
                preview.append(dict(info["samples"][0], clip=info["clip"], label=info["label"]))
            if len(buf) >= args.shard:
                save_shard(out / f"real_{args.offset:06d}_{shard_no:03d}.npz", buf)
                total += len(buf)
                shard_no += 1
                buf = []
            if k % 200 == 0:
                el = time.time() - t0
                done = total + len(buf)
                print(f"  {k}/{len(clips)} 클립, 샘플 {done}개 ({el:.0f}s, 남은 시간 약 {el / k * (len(clips) - k) / 60:.0f}분)")
    if buf:
        save_shard(out / f"real_{args.offset:06d}_{shard_no:03d}.npz", buf)
        total += len(buf)
        shard_no += 1

    with open(out / f"summary_{args.offset:06d}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    save_preview(out / "preview.png", preview)

    used = sum(1 for s in stats if s["samples"])
    errors = sum(1 for s in stats if s["error"])
    print(f"\n[결과] 샘플 {total}개, 샘플이 나온 클립 {used}/{len(stats)} ({used / max(1, len(stats)):.0%}), "
          f"에러로 건너뛴 클립 {errors}개")
    by = Counter()
    for s in stats:
        by[s["label"]] += s["samples"]
    print("동작별 샘플 수:")
    for lab, n in by.most_common():
        print(f"  {n:>6}  {lab}")
    print(f"\n[저장] {out}  (npz {shard_no}개, summary, preview.png)")
    print(f"총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
