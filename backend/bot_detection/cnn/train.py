"""CNN 3단계: 진짜(real_*.npz) vs 가짜(fake_*.npz) 분류 CNN 학습. Kaggle 노트북용 (GPU 권장), 레포 코드에 의존하지 않음.

입력: 흑백 64×64 × 10장을 10채널로 겹친 것 (05~07강 2D CNN)
모델: Conv 블록 3개 (32→64→128) + 전역 평균 풀링 + 출력 2 (0 = 진짜, 1 = 가짜)
증강 (진짜·가짜 똑같이): 좌우 반전, 시간 순서 뒤집기, 밝기·대비, 노이즈, JPEG 재압축
  → "압축 흔적·밝기" 같은 꼼수 대신 "손이 어떻게 움직이나"를 보게 하려고
평가 (검증 세트 = 클립 id로 미리 나눈 것)
  - 정확도, 진짜를 가짜로 본 비율(사람 오판), 가짜 검출률 (S1형/S4형 따로)
  - 보수적 기준: 사람 오판 ≤ 1%가 되는 문턱에서 가짜 검출률

Kaggle 사용법 (Settings → Accelerator → GPU)
  !python -u train.py
  빠른 확인: !python -u train.py --quick
결과: /kaggle/working/cnn/model.pt, curve.png, report.txt
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEQ_LEN, SIZE = 10, 64
SKIP_DIRS = {"train", "validation", "test"}


# ---------------------------------------------------------------- 데이터

def find_npz(roots: list[str], prefix: str) -> list[Path]:
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


def load(files: list[Path], is_fake: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x (n,10,64,64) uint8, split (n,), kind (n,): 진짜 = -1, 가짜 = 0(S1형)/1(S4형))."""
    xs, sp, kd = [], [], []
    for f in files:
        d = np.load(f)
        xs.append(d["x"])
        sp.append(d["split"])
        kd.append(d["kind"].astype(np.int8) if is_fake else np.full(len(d["x"]), -1, np.int8))
    if not xs:
        return np.empty((0, SEQ_LEN, SIZE, SIZE), np.uint8), np.empty(0, np.uint8), np.empty(0, np.int8)
    return np.concatenate(xs), np.concatenate(sp), np.concatenate(kd)


class Clips(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool, p_jpeg: float = 0.5):
        self.x, self.y, self.augment, self.p_jpeg = x, y, augment, p_jpeg

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        if self.augment:
            rng = np.random.default_rng()
            if rng.random() < self.p_jpeg:            # JPEG 재압축 (진짜·가짜 둘 다)
                q = int(rng.integers(40, 96))
                x = np.stack([cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1],
                                           cv2.IMREAD_GRAYSCALE) for f in x])
            x = x.astype(np.float32)
            if rng.random() < 0.5:
                x = x[:, :, ::-1]                       # 좌우 반전
            if rng.random() < 0.5:
                x = x[::-1]                             # 시간 순서 뒤집기 (진짜 손은 거꾸로 돌려도 진짜)
            x = x * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)   # 대비·밝기
            if rng.random() < 0.5:
                x = x + rng.normal(0, rng.uniform(1, 6), x.shape)  # 노이즈
            x = np.clip(x, 0, 255)
        else:
            x = x.astype(np.float32)
        x = (np.ascontiguousarray(x) / 255.0 - 0.5) / 0.25
        return torch.from_numpy(x.astype(np.float32)), int(self.y[i])


# ---------------------------------------------------------------- 모델

def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class HandNet(nn.Module):
    """(B, 10, 64, 64) → (B, 2). 10장을 채널로 겹쳐서, 첫 층부터 같은 위치의 시간 변화를 봄."""

    def __init__(self, in_ch: int = SEQ_LEN, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(block(in_ch, 32), block(32, 64), block(64, 128))   # 64 → 32 → 16 → 8
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------- 평가

@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    """가짜일 확률."""
    model.eval()
    out = []
    for x, _ in loader:
        out.append(F.softmax(model(x.to(device, non_blocking=True)), 1)[:, 1].float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def report(p: np.ndarray, y: np.ndarray, kind: np.ndarray, thr: float = 0.5) -> dict:
    real, fake = p[y == 0], p[y == 1]
    r = dict(acc=float(((p > thr) == (y == 1)).mean()),
             human_err=float((real > thr).mean()) if len(real) else float("nan"),      # 사람 오판
             fake_det=float((fake > thr).mean()) if len(fake) else float("nan"))
    for k, name in ((0, "S1"), (1, "S4")):
        m = kind == k
        r[f"det_{name}"] = float((p[m] > thr).mean()) if m.any() else float("nan")
    return r


def conservative(p: np.ndarray, y: np.ndarray, kind: np.ndarray, max_human_err: float = 0.01) -> tuple[float, dict]:
    """사람 오판 ≤ max_human_err가 되는 가장 낮은 문턱과 그때 성적."""
    real = np.sort(p[y == 0])
    if not len(real):
        return 0.5, report(p, y, kind)
    thr = float(real[min(len(real) - 1, int(np.ceil((1 - max_human_err) * len(real))) - 1)])
    return thr, report(p, y, kind, thr)


# ---------------------------------------------------------------- 메인

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/working,/kaggle/input", help="real_*/fake_*.npz 찾을 곳 (쉼표 구분)")
    ap.add_argument("--out", default="/kaggle/working/cnn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--quick", action="store_true", help="빠른 확인: 샘플 512개, 2에폭")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    roots = args.data.split(",")
    rx, rs, rk = load(find_npz(roots, "real_"), False)
    fx, fs, fk = load(find_npz(roots, "fake_"), True)
    if not len(rx) or not len(fx):
        raise SystemExit(f"데이터 부족: 진짜 {len(rx)}개, 가짜 {len(fx)}개. 1·2단계 결과 위치 확인")
    x = np.concatenate([rx, fx])
    y = np.r_[np.zeros(len(rx), np.int64), np.ones(len(fx), np.int64)]
    split = np.r_[rs, fs]
    kind = np.r_[rk, fk]
    if args.quick:
        idx = np.random.permutation(len(x))[:512]
        x, y, split, kind = x[idx], y[idx], split[idx], kind[idx]
        args.epochs = 2
    tr, va = split == 0, split == 1
    if not va.any():                                    # 검증 세트가 비면 (아주 작은 테스트) 일부를 떼어 씀
        va = np.random.rand(len(x)) < 0.2
        tr = ~va
    print(f"[데이터] 진짜 {len(rx)} / 가짜 {len(fx)} (S1형 {int((fk == 0).sum())}, S4형 {int((fk == 1).sum())}) "
          f"→ 학습 {int(tr.sum())}, 검증 {int(va.sum())}, 장치 {device}")

    train_dl = DataLoader(Clips(x[tr], y[tr], augment=True), batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(Clips(x[va], y[va], augment=False), batch_size=256, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")

    model = HandNet().to(device)
    print(f"[모델] 파라미터 {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, args.epochs * len(train_dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    hist, best, best_state = [], -1.0, None
    yv, kv = y[va], kind[va]
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        p = predict(model, val_dl, device)
        r = report(p, yv, kv)
        bal = (1 - r["human_err"] + r["fake_det"]) / 2         # 균형 정확도 (진짜·가짜 수가 달라도 공정)
        hist.append(dict(ep=ep, loss=tot / max(1, n), **r))
        print(f"  ep {ep:>2} loss {tot / max(1, n):.3f} | 검증 정확도 {r['acc']:.3f}  사람 오판 {r['human_err']:.3f}  "
              f"가짜 검출 {r['fake_det']:.3f} (S1 {r['det_S1']:.3f} / S4 {r['det_S4']:.3f})  {time.time() - t0:.0f}s")
        if bal > best:
            best = bal
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    p = predict(model, val_dl, device)
    r50 = report(p, yv, kv)
    thr, rc = conservative(p, yv, kv)
    torch.save(dict(state_dict=best_state, in_ch=SEQ_LEN, size=SIZE, norm=(0.5, 0.25),
                    threshold_1pct=thr, classes=("real", "fake")), out / "model.pt")

    lines = [
        f"검증 {int(va.sum())}개 (진짜 {int((yv == 0).sum())}, 가짜 {int((yv == 1).sum())})",
        f"문턱 0.5      : 정확도 {r50['acc']:.3f}, 사람 오판 {r50['human_err']:.3f}, 가짜 검출 {r50['fake_det']:.3f} "
        f"(S1 {r50['det_S1']:.3f} / S4 {r50['det_S4']:.3f})",
        f"문턱 {thr:.3f} (사람 오판 ≤1%): 사람 오판 {rc['human_err']:.3f}, 가짜 검출 {rc['fake_det']:.3f} "
        f"(S1 {rc['det_S1']:.3f} / S4 {rc['det_S4']:.3f})",
    ]
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n[결과]\n" + "\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ep = [h["ep"] for h in hist]
        ax[0].plot(ep, [h["loss"] for h in hist])
        ax[0].set_title("train loss")
        ax[0].set_xlabel("epoch")
        for key, lab in (("acc", "val acc"), ("human_err", "human error"), ("det_S1", "S1 det"), ("det_S4", "S4 det")):
            ax[1].plot(ep, [h[key] for h in hist], label=lab)
        ax[1].set_title("validation")
        ax[1].set_xlabel("epoch")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out / "curve.png", dpi=120)
    except Exception as exc:  # noqa: BLE001 — 그래프는 없어도 됨
        print(f"[그래프] 생략: {exc}")
    print(f"\n[저장] {out}/model.pt, report.txt, curve.png  (총 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
