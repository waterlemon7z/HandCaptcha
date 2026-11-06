# Verification test
```bash
# 기존 100개 OWLv2 실행 결과의 실패 박스를 분석
python -m cv.verification.test.verification_experiment failures

# 100개 seed × 크기 3단계 × 가림 3단계 = 900개 이미지
python -m cv.verification.test.verification_experiment compare --samples 100

# 검출을 다시 실행하지 않고 결과 표/그래프 갱신
python -m cv.verification.test.verification_experiment summarize
```

출력 기본 경로는 `verification_results/failures`, `verification_results/conditions`이다.
기존 조건 실험 CSV가 있는 폴더에는 덮어쓰지 않는다. 재실행 시 `--output-dir`로 새 폴더를 지정한다.
빠른 실행 확인은 `compare --samples 1 --output-dir /tmp/conditions-check`로 할 수 있다.

- 실패 분석: 기하 규칙별 CSV, 검토용 이미지, 분류 그래프.
- 조건 실험: 모든 모델에 동일한 이미지와 참조 대상을 전달한다. 검출기는 정답을 받지 않는다.
- 가림은 대상의 불투명 픽셀 수를 기준으로 배경을 복원하여 구현한다.
- 조건마다 seed 100개를 평가한다. 동일 seed의 9조건은 서로 독립 표본이 아니다.
- 주 정답 판정은 생성기의 `_hit`, 보조 지표는 확장하지 않은 정답 박스와 IoU≥0.5다.
- ORB·SIFT·OWLv2는 각 후보 최대 1개다. ORB/SIFT 합집합은 최대 2개인 앙상블이다.
- 배경에 대상 하나만 추가하는 통제 실험으로, 일반 생성기의 기존 정답률과 직접 비교하지 않는다.
- 모델 설정은 사전에 고정한다. 이 결과로 튜닝한다면 이후 별도 seed에서 다시 평가해야 한다.

프로젝트 의존성에 그래프용 `matplotlib`을 포함했다. OWLv2 최초 실행 시 모델을 다운로드한다.
