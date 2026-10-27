"""물체 변형과 배치."""
import cv2
import numpy as np

CLEARANCE_PX = 15        # 물체 자리로 보는 범위를 물체 둘레로 넓히는 픽셀 수. 물체끼리도 이만큼 띄운다
SURROUND_PX = 60         # 물체 주변으로 보는 고리의 바깥 반경(px)
FLOOR_COLOR_DIST = 20.0  # 바닥 색(Lab)과 이 거리 안이면 바닥으로 본다
FLOOR_SHADOW = 0.35      # 색은 바닥과 같고 바닥 밝기의 이 비율까지 어두운 곳은 그림자로 보고 바닥에 넣는다
CUT_EDGE_RATIO = 0.7     # 물체가 잘리는 선 중 이 비율 이상이 실제 윤곽선 위에 있어야 가려진 것으로 인정한다


def fit_background(img, size):
    """가운데를 잘라 size(w, h) 비율에 맞춘 뒤 크기를 맞춘다."""
    w, h = size
    ih, iw = img.shape[:2]
    if iw * h > ih * w:
        cw = ih * w // h
        x0 = (iw - cw) // 2
        img = img[:, x0:x0 + cw]
    else:
        ch = iw * h // w
        y0 = (ih - ch) // 2
        img = img[y0:y0 + ch]
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def clutter_map(bg):
    """배경의 윤곽선 지도(0/1 float). 다른 물건의 테두리에서 1이고, 빈 바닥이나 나뭇결처럼 약한 무늬는 0에 가깝다."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    return cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0


def floor_map(bg):
    """배경에서 가장 흔한 색(책상·바닥)과 비슷한 곳은 1, 사진 속 물건은 0인 지도(float 0~1).

    밝기는 그림자·나뭇결로 크게 흔들리므로 색 차이의 절반만 반영한다.
    색은 바닥과 같고 조금 어둡기만 한 곳은 그림자로 보고 바닥에 넣는다. 그래야 물체가 그림자 경계에서 잘리지 않는다.
    흰 책상 위 흰 키보드나 회색 클립처럼 바닥과 색이 같거나 비슷한 물건은 바닥으로 잡힌다.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bg, (0, 0), 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    q = (lab // np.array([16, 8, 8], np.float32)).astype(np.int32)
    key = (q[..., 0] * 32 + q[..., 1]) * 32 + q[..., 2]
    values, counts = np.unique(key, return_counts=True)
    ref = lab[key == values[np.argmax(counts)]].mean(axis=0)
    d_color = np.sqrt(((lab[..., 1:] - ref[1:]) ** 2).sum(-1))
    d_light = lab[..., 0] - ref[0]
    same = np.sqrt(d_color ** 2 + (0.5 * d_light) ** 2) < FLOOR_COLOR_DIST
    shadow = (d_color < 0.6 * FLOOR_COLOR_DIST) & (d_light < 0) & (lab[..., 0] > ref[0] * (1 - FLOOR_SHADOW))
    mask = (same | shadow).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    # 가려지는 경계가 계단처럼 보이지 않게 살짝 흐린다
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 1.0)


def transform_object(obj_bgra, rng, long_side, angle_range):
    """물체를 긴 변이 long_side(px)가 되게 줄이고, angle_range(도) 안의 랜덤 각도로 회전한다.

    회전해도 잘리지 않게 캔버스를 넓히며, 바깥은 투명(alpha 0)이다.
    검수 쪽은 원본 물체 이미지로 찾으므로, 붙이는 모양을 원본과 다르게 만드는 단계다.
    """
    oh, ow = obj_bgra.shape[:2]
    f = long_side / max(oh, ow)
    # 투명 픽셀의 색은 의미가 없으므로 alpha를 곱해 둔 상태로 보간해야 테두리에 엉뚱한 색이 섞이지 않는다
    a = obj_bgra[..., 3:].astype(np.float32) / 255.0
    pm = np.dstack([obj_bgra[..., :3] * a, a[..., 0] * 255.0])
    pm = cv2.resize(pm, (max(1, round(ow * f)), max(1, round(oh * f))),
                    interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)

    h, w = pm.shape[:2]
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), rng.uniform(*angle_range), 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    # 테두리 1px은 항상 투명하게 남겨, 회전한 모서리가 캔버스 끝에 걸리지 않게 한다
    nw, nh = int(np.ceil(w * cos + h * sin)) + 2, int(np.ceil(w * sin + h * cos)) + 2
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    pm = cv2.warpAffine(pm, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    alpha = pm[..., 3:] / 255.0
    rgb = np.where(alpha > 0, pm[..., :3] / np.maximum(alpha, 1e-6), 0)
    return np.clip(np.rint(np.dstack([rgb, pm[..., 3]])), 0, 255).astype(np.uint8)


def _dilated(alpha, pad, radius):
    """alpha(uint8)를 사방 pad만큼 넓힌 캔버스에 놓고 radius(px)만큼 팽창한 0/1 float 지도."""
    canvas = cv2.copyMakeBorder(alpha, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    k = 2 * radius + 1
    return (cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 127).astype(np.float32)


def place_object(bg, obj_bgra, rng, margin, n_candidates=1, top_k=1, occupied=None, floor=None, hidden_range=None):
    """크기·회전이 정해진 물체를, 자리는 비어 있고 주변은 복잡한 곳에 붙인다 (숨은그림찾기).

    margin은 가장자리에서 띄울 비율이다.
    위치는 후보 n_candidates개를 뽑아 점수를 매기고, 점수가 좋은 top_k개 중 하나를 고른다.
    점수 = 3 × 물체 자리의 윤곽선 밀도 - 주변 고리의 윤곽선 밀도 (낮을수록 좋음).
    자리에 윤곽선이 있으면 다른 물건 위에 떠 보이고, 주변이 비어 있으면 물체가 혼자 눈에 띄기 때문이다.
    occupied(H×W bool)는 이미 놓은 물체 자리로, CLEARANCE_PX 안으로 겹치는 후보는 고르지 않고
    주변의 복잡도로도 세지 않는다 (숨긴 물체끼리 모이지 않게).
    floor(floor_map 결과)와 hidden_range((최소, 최대) 비율)를 주면, 물체가 사진 속 물건 밑에
    그 비율만큼 깔리는 후보만 고르고 바닥 위 부분만 그린다. 잘리는 선이 실제 물건의 윤곽선 위에 있지 않으면
    (바닥 얼룩·그림자 경계에서 잘리면) 가려진 게 아니라 잘린 것처럼 보이므로 고르지 않는다.
    그런 후보가 없으면 가리지 않고 놓는다.
    반환: (alpha로 합성한 이미지 uint8, 배경 크기의 alpha float32 0~1). alpha는 보이는 부분만이다.
    """
    h, w = bg.shape[:2]
    oh, ow = obj_bgra.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    xs = rng.integers(mx, w - mx - ow + 1, size=n_candidates)
    ys = rng.integers(my, h - my - oh + 1, size=n_candidates)

    # 매끈한 물건은 안쪽에 윤곽선이 없으므로, 물체 자리도 둘레 CLEARANCE_PX까지 넓혀서 다른 물건의 테두리가 걸리는지 본다
    r = SURROUND_PX
    inner = _dilated(obj_bgra[..., 3], r, CLEARANCE_PX)
    ring = _dilated(obj_bgra[..., 3], r, r) - inner
    edges = clutter_map(bg)
    outline = edges > 0
    if occupied is not None:
        edges[occupied] = 0
    edges = cv2.copyMakeBorder(edges, r, r, r, r, cv2.BORDER_CONSTANT, value=0)
    windows = [edges[y:y + oh + 2 * r, x:x + ow + 2 * r] for x, y in zip(xs, ys)]
    # 자리 쪽에 3배를 줘서, 빽빽한 곳 한가운데(둘 다 높음)나 물건에 걸친 자리가 빈 바닥보다 좋은 점수를 받지 않게 한다
    scores = np.array([3 * (win * inner).sum() / inner.sum() - (win * ring).sum() / ring.sum() for win in windows])
    if occupied is not None:
        busy = cv2.copyMakeBorder(occupied.astype(np.uint8), r, r, r, r, cv2.BORDER_CONSTANT, value=0)
        overlap = [(busy[y:y + oh + 2 * r, x:x + ow + 2 * r] * inner).any() for x, y in zip(xs, ys)]
        scores[overlap] = np.inf
    occlude = False
    if floor is not None:
        solid = obj_bgra[..., 3] > 127
        hidden = np.array([1 - (floor[y:y + oh, x:x + ow][solid] > 0.5).mean() for x, y in zip(xs, ys)])
        tucked = (hidden >= hidden_range[0]) & (hidden <= hidden_range[1])
        # 잘리는 선 = 보이는 부분 중 가려진 부분과 맞닿은 픽셀. 윤곽선에서 2px 안이면 윤곽선 위로 본다
        near = cv2.dilate(outline.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        touch = np.ones((3, 3), np.uint8)
        for i in np.flatnonzero(tucked):
            x, y = xs[i], ys[i]
            on_floor = floor[y:y + oh, x:x + ow] > 0.5
            hid = (solid & ~on_floor).astype(np.uint8)
            cut = solid & on_floor & (cv2.dilate(hid, touch) > 0)
            tucked[i] = np.count_nonzero(cut & near[y:y + oh, x:x + ow]) >= CUT_EDGE_RATIO * np.count_nonzero(cut)
        occlude = bool((tucked & np.isfinite(scores)).any())
        if occlude:
            scores[~tucked] = np.inf
    order = np.argsort(scores, kind="stable")
    # 겹치지 않는 후보만 고른다. 모든 후보가 겹치는 경우는 사실상 없지만, 그때는 가장 나은 후보로 둔다
    best = order[:max(1, min(top_k, int(np.isfinite(scores).sum())))]
    pick = best[rng.integers(len(best))]
    x0, y0 = int(xs[pick]), int(ys[pick])

    alpha = np.zeros((h, w), np.float32)
    alpha[y0:y0 + oh, x0:x0 + ow] = obj_bgra[..., 3] / 255.0
    if occlude:
        # 사진 속 물건 위에 걸친 부분은 그리지 않아, 그 물건 밑에 깔린 것처럼 보이게 한다
        alpha[y0:y0 + oh, x0:x0 + ow] *= floor[y0:y0 + oh, x0:x0 + ow]
    fg = bg.astype(np.float32)
    # 투명 픽셀의 색은 의미가 없으므로 alpha로 먼저 섞어 테두리 색 번짐을 막는다
    a = alpha[y0:y0 + oh, x0:x0 + ow, None]
    fg[y0:y0 + oh, x0:x0 + ow] = obj_bgra[..., :3] * a + fg[y0:y0 + oh, x0:x0 + ow] * (1 - a)
    return np.clip(np.rint(fg), 0, 255).astype(np.uint8), alpha
