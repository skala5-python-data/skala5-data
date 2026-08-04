# Day 2 End-to-End 데이터 분석 결과 보고서

## 1. 프로젝트 개요

Adult Census Income 데이터를 정제하고 시각화한 뒤, 두 소득 집단의 주당 근무시간 차이를 검정했습니다. 개인 특성을 이용해 연 소득이 50K를 초과하는지 예측하는 분류 모델도 학습했습니다.

## 2. 데이터 처리 결과

| 항목 | Pandas | Polars |
|---|---:|---:|
| 정제 후 행 수 | {{ comparison.pandas.rows }} | {{ comparison.polars.rows }} |
| 컬럼 수 | {{ comparison.pandas.columns }} | {{ comparison.polars.columns }} |
| 로딩 시간(초) | {{ comparison.pandas.load_seconds }} | {{ comparison.polars.load_seconds }} |
| 추정 메모리(Byte) | {{ comparison.pandas.memory_bytes }} | {{ comparison.polars.memory_bytes }} |
| 제거한 중복 행 | {{ comparison.pandas.duplicates_removed }} | {{ comparison.polars.duplicates_removed }} |

- Pandas와 Polars 정제 결과의 크기 일치 여부: `{{ comparison.same_cleaned_shape }}`
- 전체 정제 후 행 수: {{ eda.rows }}

## 3. 소득 분포

{% for label, count in eda.income_counts.items() %}
- {{ label }}: {{ count }}명
{% endfor %}

![Income distribution](../charts/income_distribution.png)

## 4. 학력 수준별 고소득 비율

| 학력 그룹 | 표본 수 | 고소득 비율 |
|---|---:|---:|
{% for label, result in eda.education_income_summary.items() %}
| {{ label }} | {{ result.count }} | {{ result.high_income_rate }}% |
{% endfor %}

![High-income rate by education level](../charts/education_income_rate.png)

## 5. 학력과 직업별 고소득 비율

표본이 30명 이상인 학력·직업 조합만 히트맵에 표시했습니다.

![High-income rate by education and occupation](../charts/education_occupation_income_heatmap.png)

## 6. 상관관계

![Correlation heatmap](../charts/correlation_heatmap.png)

인터랙티브 차트는 [hours_by_income.html](../charts/hours_by_income.html) 파일에서 확인합니다.

## 7. 독립표본 t-test

- 검정 방법: {{ statistics.test }}
- 분석 변수: {{ statistics.variable }}
- `<=50K` 집단 평균: {{ statistics.low_income_mean }}
- `>50K` 집단 평균: {{ statistics.high_income_mean }}
- t통계량: {{ statistics.t_statistic }}
- p-value: {{ statistics.p_value_display }}
- 유의수준: {{ statistics.alpha }}

{{ statistics.interpretation }}

## 8. 머신러닝 Pipeline 결과

- 모델: {{ model.model }}
- 학습 데이터: {{ model.train_rows }}행
- 테스트 데이터: {{ model.test_rows }}행
- 정확도: {{ model.accuracy }}
- 정밀도: {{ model.precision }}
- 재현율: {{ model.recall }}
- F1 점수: {{ model.f1 }}
- 혼동행렬: {{ model.confusion_matrix }}
- 저장 후 재로딩 예측 일치: `{{ model.reloaded_predictions_match }}`

![Model metrics](../charts/model_metrics.png)

![Confusion matrix](../charts/confusion_matrix.png)

## 9. 산출물

- 정제 CSV: `data/processed/adult_cleaned.csv`
- 정제 Parquet: `data/processed/adult_cleaned.parquet`
- 통계 결과: `outputs/metrics/statistics.json`
- 모델 성능: `outputs/metrics/model_metrics.json`
- 저장 모델: `outputs/models/income_pipeline.joblib`
