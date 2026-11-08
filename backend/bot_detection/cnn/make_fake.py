"""CNN 2단계: 1단계 진짜 샘플마다 짝이 되는 '가짜 손' 샘플을 만든다. Kaggle 노트북용, 레포 코드에 의존하지 않음.

가짜 1개 = 우리 합성 봇(synth.py S1/S4)과 같은 방식
- 손 사진: 진짜 샘플의 원본 클립에서 프레임 한 장을 골라, 랜드마크 convexHull 마스크로 손만 오려냄.
  마스크 경계는 가우시안으로 부드럽게 (03강 블렌딩)
- 배경: 다른 클립(No gesture = 손 없는 진짜 웹캠 영상)의 연속 10장
- 움직임: 진짜 샘플의 검지 끝 경로를 그대로 따라 손 사진을 옮김.
  S1형 = 옮기기만 / S4형 = 매 장 조금씩 회전·확대 (warpAffine, 02-2)
- crop: 진짜 샘플과 같은 경로·같은 크기로 검지 끝 중심 crop → 64×64 흑백
  → 진짜와 가짜는 경로·crop 크기·손 모양이 같고, "손이 통째로 움직이나"만 다름
- 꼼수 차단: 합성한 프레임을 JPEG로 다시 압축 (Jester 원본처럼 압축 흔적이 있게). 일부는 센서 노이즈 추가(S3형)

Kaggle 사용법 (1단계와 같은 노트북 또는 1단계 결과를 Input으로 붙인 노트북)
  !python -u make_fake.py
결과: /kaggle/working/jester_fake/fake_*.npz, preview.png (윗줄 진짜, 아랫줄 가짜)
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

WRIST, MIDDLE_MCP, INDEX_TIP = 0, 9, 8
SEQ_LEN = 10
SIZE = 64
SKIP_DIRS = {"train", "validation", "test"}     # Jester 클립 폴더 목록 (수만 개)은 뒤지지 않음


# ---------------------------------------------------------------- 파일 찾기

def find_npz(roots: list[str], prefix: str) -> list[Path]:
    """roots 아래의 prefix*.npz. Jester 클립 폴더 목록 안으로는 들어가지 않는다."""
    out = set()
    for r in roots:
        if not os.path.isdir(r):
            continue
        for dirpath, dirnames, filenames in os.walk(r):
            dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS]
            if len(dirnames) > 500:
                dirnames[:] = []
            out.update(Path(dirpath) / f for f in filenames if f.startswith(prefix) and f.endswith(".npz"))
    return sorted(out)


def find_train_dir(root: Path) -> Path | None:
    """하위 폴더가 1000개 이상인 폴더 (Jester Train)."""
    level = [root]
    for _ in range(5):
        found, nxt = [], []
        for d in level:
            try:
                subs = [Path(e.path) for e in os.scandir(d) if e.is_dir()]
            except OSError:
                continue
            if len(subs) >= 1000:
                found.append((d, len(subs)))
            else:
                nxt.extend(subs)
        if found:
            return max(found, key=lambda x: (x[0].name.lower() == "train", x[1]))[0]
        level = nxt
    return None


def load_labels(root: Path) -> dict[str, str]:
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


def jpgs(clip: Path) -> list[Path]:
    return sorted(p for p in clip.iterdir() if p.suffix.lower() == ".jpg")


# ---------------------------------------------------------------- 합성

def crop_at(frame: np.ndarray, center, side: float) -> np.ndarray:
    """center 중심 side×side crop, 화면 밖은 검정. center가 화면 밖 멀리 있어도 항상 side×side."""
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


def hand_mask(shape, lm: np.ndarray, hand: float) -> np.ndarray:
    """랜드마크 convexHull → 조금 넓히고(손가락 두께) 경계를 가우시안으로 부드럽게. 0~1 float."""
    mask = np.zeros(shape[:2], np.uint8)
    hull = cv2.convexHull(np.round(lm).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    r = max(1, int(round(hand * 0.12)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    sigma = max(0.5, hand * 0.05)
    return cv2.GaussianBlur(mask, (0, 0), sigma).astype(np.float32) / 255.0


def pose_walk(rng, kind: int, n: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """매 장의 회전(도)·확대 배율. S1형은 0도·1배 고정, S4형은 조금씩 랜덤하게 변함."""
    if kind == 0:
        return np.zeros(n), np.ones(n)
    th = np.clip(np.cumsum(np.r_[0, rng.normal(0, cfg["rot_step"], n - 1)]), -cfg["rot_max"], cfg["rot_max"])
    sc = np.clip(np.exp(np.cumsum(np.r_[0, rng.normal(0, cfg["scale_step"], n - 1)])),
                 1 - cfg["scale_max"], 1 + cfg["scale_max"])
    return th, sc


def warp_to(tip0, tip, theta_deg: float, scale: float) -> np.ndarray:
    """손 사진의 검지 끝(tip0)을 tip으로 옮기고, 그 점을 중심으로 회전·확대하는 2×3 행렬."""
    t = np.deg2rad(theta_deg)
    a, b = scale * np.cos(t), scale * np.sin(t)
    tx = tip[0] - (a * tip0[0] - b * tip0[1])
    ty = tip[1] - (b * tip0[0] + a * tip0[1])
    return np.float32([[a, -b, tx], [b, a, ty]])


_CFG: dict = {}
_BG: list[str] = []


def init_worker(cfg: dict, bg_clips: list[str]) -> None:
    global _CFG, _BG
    _CFG, _BG = cfg, bg_clips


def make_one(task) -> dict | None:
    """가짜 하나. 에러가 나면 None (전체 실행은 계속)."""
    try:
        return _make_one(task)
    except Exception:  # noqa: BLE001
        return None


def _make_one(task) -> dict | None:
    i, clip, start, lms, side, hand, split, seed = task
    cfg = _CFG
    rng = np.random.default_rng(seed)
    src_files = jpgs(Path(clip))
    j0 = int(rng.integers(0, SEQ_LEN))                  # 손 사진으로 쓸 프레임 (10장 중 하나)
    if start + j0 >= len(src_files):
        return None
    src = cv2.imread(str(src_files[start + j0]))
    if src is None:
        return None
    h, w = src.shape[:2]

    # 배경: No gesture 클립의 연속 10장
    for _ in range(5):
        bg_clip = Path(_BG[int(rng.integers(0, len(_BG)))])
        bg_files = jpgs(bg_clip)
        if len(bg_files) >= SEQ_LEN:
            break
    else:
        return None
    b0 = int(rng.integers(0, len(bg_files) - SEQ_LEN + 1))
    bgs = [cv2.imread(str(p)) for p in bg_files[b0:b0 + SEQ_LEN]]
    if any(b is None for b in bgs):
        return None
    bgs = [b if b.shape[:2] == (h, w) else cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA) for b in bgs]

    lm0 = lms[j0]
    tip0 = lm0[INDEX_TIP]
    tips = lms[:, INDEX_TIP]
    mask = hand_mask(src.shape, lm0, hand)
    kind = int(rng.random() < cfg["p_s4"])              # 0 = S1형(옮기기만), 1 = S4형(회전·확대)
    th, sc = pose_walk(rng, kind, SEQ_LEN, cfg)
    noise = float(rng.uniform(1.5, 5.0)) if rng.random() < cfg["p_noise"] else 0.0
    q = int(rng.integers(cfg["q_min"], cfg["q_max"] + 1))

    x = []
    srcf = src.astype(np.float32)
    for t in range(SEQ_LEN):
        m = warp_to(tip0, tips[t], th[t], sc[t])
        hand_img = cv2.warpAffine(srcf, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        comp = alpha * hand_img + (1 - alpha) * bgs[t].astype(np.float32)
        if noise:
            comp += rng.normal(0, noise, comp.shape)
        comp = np.clip(comp, 0, 255).astype(np.uint8)
        comp = cv2.imdecode(cv2.imencode(".jpg", comp, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
        x.append(to_sample(crop_at(comp, tips[t], side)))
    return dict(i=i, x=np.stack(x), clip=Path(clip).name, bg_clip=bg_clip.name, start=start,
                kind=kind, noise=noise, q=q, split=split)


# ---------------------------------------------------------------- 저장

def save_shard(path: Path, buf: list[dict]) -> None:
    np.savez_compressed(
        path,
        x=np.stack([b["x"] for b in buf]).astype(np.uint8),        # (n, 10, 64, 64)
        real_index=np.array([b["i"] for b in buf], dtype=np.int64),   # 짝이 되는 진짜 샘플 번호 (real 파일 순서대로 이어 붙인 기준)
        clip=np.array([b["clip"] for b in buf]),                    # 손 사진을 가져온 클립
        bg_clip=np.array([b["bg_clip"] for b in buf]),              # 배경 클립
        start=np.array([b["start"] for b in buf], dtype=np.int32),
        kind=np.array([b["kind"] for b in buf], dtype=np.uint8),      # 0 = S1형, 1 = S4형
        noise=np.array([b["noise"] for b in buf], dtype=np.float32),  # 0이면 노이즈 없음
        q=np.array([b["q"] for b in buf], dtype=np.uint8),            # JPEG 품질
        split=np.array([b["split"] for b in buf], dtype=np.uint8),    # 진짜 짝과 같은 분리
    )


def save_preview(path: Path, pairs: list[tuple[np.ndarray, dict]]) -> None:
    """샘플마다 두 줄: 윗줄 진짜, 아랫줄 가짜 (2배 확대)."""
    def row(x, text):
        line = np.hstack([cv2.copyMakeBorder(f, 0, 2, 0, 2, cv2.BORDER_CONSTANT, value=0) for f in x])
        line = cv2.resize(line, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        line = cv2.copyMakeBorder(cv2.cvtColor(line, cv2.COLOR_GRAY2BGR), 20, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(line, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return line
    lines = []
    for real_x, f in pairs:
        lines.append(row(real_x, f"REAL {f['clip']}"))
        lines.append(row(f["x"], f"FAKE {'S4' if f['kind'] else 'S1'} noise={f['noise']:.1f} q={f['q']} bg={f['bg_clip']}"))
        lines.append(np.zeros((8, lines[-1].shape[1], 3), np.uint8))
    if lines:
        cv2.imwrite(str(path), np.vstack(lines))


# ---------------------------------------------------------------- 메인

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input", help="Jester 위치")
    ap.add_argument("--real", default="/kaggle/working,/kaggle/input", help="real_*.npz를 찾을 곳 (쉼표 구분)")
    ap.add_argument("--out", default="/kaggle/working/jester_fake")
    ap.add_argument("--max", type=int, default=0, help="만들 가짜 수 (0 = 진짜 샘플 수만큼)")
    ap.add_argument("--p-s4", type=float, default=0.7, help="S4형(회전·확대) 비율, 나머지는 S1형")
    ap.add_argument("--p-noise", type=float, default=0.3, help="센서 노이즈 추가 비율 (S3형)")
    ap.add_argument("--rot-step", type=float, default=2.5, help="장마다 회전 변화 표준편차 (도)")
    ap.add_argument("--rot-max", type=float, default=20.0)
    ap.add_argument("--scale-step", type=float, default=0.025)
    ap.add_argument("--scale-max", type=float, default=0.15)
    ap.add_argument("--q-min", type=int, default=60, help="JPEG 재압축 품질 범위")
    ap.add_argument("--q-max", type=int, default=95)
    ap.add_argument("--bg-label", default="No gesture")
    ap.add_argument("--shard", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    real_files = find_npz(args.real.split(","), "real_")
    if not real_files:
        raise SystemExit("real_*.npz를 못 찾음. 1단계(jester_prep.py) 결과가 /kaggle/working 또는 Input에 있는지 확인")
    print(f"[진짜] npz {len(real_files)}개: {real_files[0].parent}")

    train_dir = find_train_dir(Path(args.root))
    if train_dir is None:
        raise SystemExit(f"{args.root} 아래에서 Jester 클립 폴더를 못 찾음")
    labels = load_labels(Path(args.root)) or load_labels(train_dir.parent)
    names = {e.name for e in os.scandir(train_dir) if e.is_dir()}
    bg_clips = sorted(str(train_dir / c) for c, lab in labels.items() if lab == args.bg_label and c in names)
    print(f"[배경] '{args.bg_label}' 클립 {len(bg_clips)}개")
    if not bg_clips:
        raise SystemExit("배경 클립이 없음. --bg-label 확인")

    # 진짜 샘플 메타데이터 (x는 미리보기용 몇 개만)
    tasks, real_preview, i = [], {}, 0
    for f in real_files:
        d = np.load(f)
        n = len(d["clip"])
        for k in range(n):
            tasks.append((i, str(train_dir / str(d["clip"][k])), int(d["start"][k]), d["lms"][k],
                          float(d["side"][k]), float(d["hand"][k]), int(d["split"][k]), args.seed * 1_000_003 + i))
            if len(real_preview) < 6:
                real_preview[i] = d["x"][k]
            i += 1
    if args.max:
        rnd = random.Random(args.seed)
        keep = set(real_preview) | set(rnd.sample(range(len(tasks)), min(args.max, len(tasks))))
        tasks = [t for t in tasks if t[0] in keep][:max(args.max, len(real_preview))]
    print(f"[가짜] {len(tasks)}개 만들기, 코어 {args.workers}개")

    cfg = dict(p_s4=args.p_s4, p_noise=args.p_noise, rot_step=args.rot_step, rot_max=args.rot_max,
               scale_step=args.scale_step, scale_max=args.scale_max, q_min=args.q_min, q_max=args.q_max)
    buf, shard_no, total, fail = [], 0, 0, 0
    preview = []
    with Pool(args.workers, initializer=init_worker, initargs=(cfg, bg_clips)) as pool:
        for k, r in enumerate(pool.imap_unordered(make_one, tasks, chunksize=16), 1):
            if r is None:
                fail += 1
            else:
                buf.append(r)
                if r["i"] in real_preview:
                    preview.append((real_preview[r["i"]], r))
            if len(buf) >= args.shard:
                save_shard(out / f"fake_{shard_no:03d}.npz", buf)
                total += len(buf)
                shard_no += 1
                buf = []
            if k % 1000 == 0:
                el = time.time() - t0
                print(f"  {k}/{len(tasks)} ({el:.0f}s, 남은 시간 약 {el / k * (len(tasks) - k) / 60:.0f}분)")
    if buf:
        save_shard(out / f"fake_{shard_no:03d}.npz", buf)
        total += len(buf)
        shard_no += 1
    save_preview(out / "preview.png", sorted(preview, key=lambda p: p[1]["i"]))

    print(f"\n[결과] 가짜 {total}개 (실패 {fail}개), npz {shard_no}개 → {out}")
    print(f"총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
