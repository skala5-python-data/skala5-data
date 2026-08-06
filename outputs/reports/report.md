# 학력 수준과 고소득 관계 분석 보고서

## 1. 프로젝트 개요

Adult Census Income 데이터를 바탕으로 학력 수준에 따른 고소득 비율을 분석하고, 학력과 직업을 함께 고려했을 때의 차이를 시각화했습니다. 두 소득 집단의 교육 수준 차이를 검정하고, 개인 특성을 이용해 연 소득이 50K를 초과하는지 예측하는 분류 모델도 학습했습니다.

## 2. 데이터 처리 결과

| 항목 | Pandas | Polars |
|---|---:|---:|
| 정제 후 행 수 | 32537 | 32537 |
| 컬럼 수 | 15 | 15 |
| 로딩 시간(초) | 0.026629 | 0.017768 |
| 추정 메모리(Byte) | 20722338 | 3218734 |
| 제거한 중복 행 | 24 | 24 |

- Pandas와 Polars 정제 결과의 크기 일치 여부: `True`
- 전체 정제 후 행 수: 32537

## 3. 소득 분포

- <=50K: 24698명
- >50K: 7839명

![Income distribution](../charts/income_distribution.png)

## 4. 학력 수준별 고소득 비율

| 학력 그룹 | 표본 수 | 고소득 비율 |
|---|---:|---:|
| 고졸 미만 | 4248 | 5.74% |
| 고졸 | 10494 | 15.95% |
| 대학 과정·전문학사 | 9731 | 20.68% |
| 학사 | 5353 | 41.49% |
| 대학원 이상 | 2711 | 62.26% |

![High-income rate by education level](../charts/education_income_rate.png)

## 5. 학력과 직업별 고소득 비율

표본이 30명 이상인 학력·직업 조합만 히트맵에 표시했습니다.

![High-income rate by education and occupation](../charts/education_occupation_income_heatmap.png)

## 6. 상관관계

![Correlation heatmap](../charts/correlation_heatmap.png)

## 7. 독립표본 t-test

- 검정 방법: Welch independent two-sample t-test
- 분석 변수: education-num
- `<=50K` 집단 평균: 9.5961
- `>50K` 집단 평균: 11.6122
- t통계량: -64.876078
- p-value: < 1e-300
- 유의수준: 0.05

p-value가 0.05보다 작으므로 두 소득 집단의 교육 수준 평균은 통계적으로 유의한 차이가 있다고 해석합니다.

## 8. 머신러닝 Pipeline 결과

- 모델: LogisticRegression
- 학습 데이터: 26029행
- 테스트 데이터: 6508행
- 정확도: 0.856792
- 정밀도: 0.743492
- 재현율: 0.61926
- F1 점수: 0.675713
- 혼동행렬: [[4605, 335], [597, 971]]
- 저장 후 재로딩 예측 일치: `True`

![Model metrics](../charts/model_metrics.png)

![Confusion matrix](../charts/confusion_matrix.png)

## 9. 산출물

- 정제 CSV: `data/processed/adult_cleaned.csv`
- 정제 Parquet: `data/processed/adult_cleaned.parquet`
- 통계 결과: `outputs/metrics/statistics.json`
- 모델 성능: `outputs/metrics/model_metrics.json`
- 저장 모델: `outputs/models/income_pipeline.joblib`