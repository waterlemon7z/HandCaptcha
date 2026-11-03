"""정답 정보를 사용하지 않는 ORB/SIFT 자동 풀이 검수기.

ProblemGenerator(validator=difficult_check)로 연결한다.
입력은 uint8 BGR/BGRA/흑백 배열, 출력은 문제 좌표의 xyxy 박스 리스트.
재생성, 정답 비교, 대기 루프, GUI 및 파일 입출력은 담당하지 않는다.
"""
import cv2
import numpy as np


def preprocess_image(img):
    """사용자가 작성한 전처리 함수. 잘못된 입력은 오류로 알린다."""
    if not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("이미지는 비어 있지 않은 NumPy 배열이어야 합니다.")
    if img.dtype != np.uint8:
        raise ValueError("이미지 dtype은 uint8이어야 합니다.")
    if img.ndim == 2:
        return np.ascontiguousarray(img)
    if img.ndim == 3 and img.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(img, code)
    raise ValueError("흑백, BGR, BGRA 이미지만 지원합니다.")


def _detect(scene, target, mask, method):
    if method == 'orb':
        detector = cv2.ORB_create(nfeatures=3000, edgeThreshold=15)
        norm = cv2.NORM_HAMMING
    elif method == 'sift':
        if not hasattr(cv2, 'SIFT_create'):
            raise RuntimeError('SIFT를 지원하는 OpenCV가 필요합니다.')
        detector = cv2.SIFT_create(nfeatures=3000)
        norm = cv2.NORM_L2
    else:
        raise ValueError("method는 orb, sift, auto 중 하나여야 합니다.")
    kt, dt = detector.detectAndCompute(target, mask)
    ks, ds = detector.detectAndCompute(scene, None)
    if dt is None or ds is None or len(kt) < 8 or len(ks) < 8:
        return []
    pairs = cv2.BFMatcher(norm).knnMatch(dt, ds, k=2)
    unique = {}
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            m = pair[0]
            if m.trainIdx not in unique or m.distance < unique[m.trainIdx].distance:
                unique[m.trainIdx] = m
    matches = list(unique.values())
    if len(matches) < 8:
        return []
    src = np.float32([kt[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([ks[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return []
    count = int(inliers.sum())
    if count < 6 or count / len(matches) < 0.5:
        return []
    x, y, w, h = cv2.boundingRect(mask)
    corners = np.float32([[x,y], [x+w,y], [x+w,y+h], [x,y+h]]).reshape(-1,1,2)
    # 원근 변환의 무한대 경계를 가로지르는 변환은 거부한다.
    homogeneous = np.c_[corners.reshape(-1,2), np.ones(4)] @ matrix.T
    denominators = homogeneous[:,2]
    if not (np.all(denominators > 1e-8) or np.all(denominators < -1e-8)):
        return []
    polygon = cv2.perspectiveTransform(corners, matrix)
    if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
        return []
    ih, iw = scene.shape
    area = abs(cv2.contourArea(polygon))
    if area < 16 or area > ih * iw:
        return []
    points = polygon.reshape(-1,2)
    lo = np.floor(points.min(axis=0))
    hi = np.ceil(points.max(axis=0))
    x1, y1 = np.clip(lo, [0,0], [iw,ih]).astype(int)
    x2, y2 = np.clip(hi, [0,0], [iw,ih]).astype(int)
    if x2 <= x1 or y2 <= y1:
        return []
    return [(int(x1), int(y1), int(x2), int(y2))]


def difficult_check(image, target, *, method='auto'):
    """대상 하나를 찾아 최대 한 개의 박스를 반환한다. 미검출은 [].

    auto는 ORB 미검출 시 SIFT를 시도한다. 미검출은 인간 난이도 보장이 아니다.
    BGRA 대상은 알파가 0인 영역을 제외하고 투명 여백도 잘라낸다.
    x2/y2는 슬라이스 끝 좌표이며 정답 여부는 생성기가 판단한다.
    """
    if method not in ('auto', 'orb', 'sift'):
        raise ValueError('method는 auto, orb, sift 중 하나여야 합니다.')
    scene = preprocess_image(image)
    gray = preprocess_image(target)
    mask = (np.where(target[:,:,3] > 0, 255, 0).astype(np.uint8)
            if target.ndim == 3 and target.shape[2] == 4
            else np.full(gray.shape, 255, np.uint8))
    if not mask.any():
        return []
    x,y,w,h = cv2.boundingRect(mask)
    gray = gray[y:y+h, x:x+w].copy()
    mask = mask[y:y+h, x:x+w].copy()
    # 투명 픽셀의 임의 RGB 값이 기술자에 영향을 주지 않도록 고정한다.
    gray[mask == 0] = 0
    methods = ('orb', 'sift') if method == 'auto' else (method,)
    for current in methods:
        if current == 'sift' and method == 'auto' and not hasattr(cv2, 'SIFT_create'):
            continue
        boxes = _detect(scene, gray, mask, current)
        if boxes:
            return boxes
    return []


# 이전 파일명/함수명 표기를 사용하는 코드도 연결할 수 있도록 제공
difficulty_check = difficult_check
