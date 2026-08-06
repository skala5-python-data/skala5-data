# 학력 수준과 고소득 관계 분석 보고서

## 1. 프로젝트 개요

Adult Census Income 데이터를 바탕으로 학력 수준에 따른 고소득 비율을 분석하고, 학력과 직업을 함께 고려했을 때의 차이를 시각화했습니다. 두 소득 집단의 교육 수준 차이를 검정하고, 개인 특성을 이용해 연 소득이 50K를 초과하는지 예측하는 분류 모델도 학습했습니다.

## 2. 데이터 처리 결과

| 항목 | Pandas | Polars |
|---|---:|---:|
| 정제 후 행 수 | {{ comparison.pandas.rows }} | {{ comparison.polars.rows }} |
| 컬럼 수 | {{ comparison.pandas.columns }} | {{ comparison.polars.columns }} |
| 로딩 시간(초) | {{ comparison.pandas.load_seconds }} | {{ comparison.polars.load_seconds }} |
| 추정 메모리(Byte) | {{ comparison.pandas.memory_bytes }} | {{ comparison.polars.memory_bytes }} |
| 제거한 중복 행 | {{ comparison.pandas.duplicates_removed }} | {{ comparison.polars.duplicates_removed }} |
| 처리 전 결측치 | {{ comparison.pandas.missing_values_before }} | {{ comparison.polars.missing_values_before }} |
| 처리 후 결측치 | {{ comparison.pandas.missing_values_after }} | {{ comparison.polars.missing_values_after }} |

- Pandas와 Polars 정제 결과의 크기 일치 여부: `{{ comparison.same_cleaned_shape }}`
- 전체 정제 후 행 수: {{ eda.rows }}

원본 데이터에서 `?`로 표시된 값을 결측치로 인식했습니다. 예측 대상인 `income`이 결측된 행은 제거하고, 범주형 설명변수의 결측치는 데이터 손실을 줄이기 위해 `Unknown` 범주로 대체했습니다. 머신러닝 Pipeline에도 수치형 중앙값 및 범주형 최빈값 대치 단계를 포함하여 새로운 데이터에 결측치가 들어오는 경우를 처리하도록 구성했습니다.

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

### 5.1 같은 학력에서도 직업별 고소득 비율이 다른가?

| 학력 그룹 | 고소득률이 낮은 직업 | 비율 | 고소득률이 높은 직업 | 비율 | 격차 |
|---|---|---:|---|---:|---:|
{% for education_group, result in eda.occupation_gap_by_education.items() %}
| {{ education_group }} | {{ result.lowest_occupation }} | {{ result.lowest_rate }}% | {{ result.highest_occupation }} | {{ result.highest_rate }}% | {{ result.gap_percentage_points }}%p |
{% endfor %}

모든 학력 그룹에서 직업별 고소득률 차이가 나타났습니다. 따라서 학력이 같더라도 직업에 따라 고소득 가능성이 다르게 관측된다고 해석할 수 있습니다.

![Occupation rate gap within education level](../charts/education_occupation_rate_gap.png)

### 5.2 학력에 따른 고소득률 격차가 가장 큰 직업

아래 표는 표본 30명 이상인 학력 그룹이 3개 이상 존재하는 직업을 대상으로, 관측된 최고·최저 학력 그룹의 고소득률 격차를 계산한 결과입니다.

| 순위 | 직업 | 낮은 비율의 학력 그룹 | 비율 | 높은 비율의 학력 그룹 | 비율 | 격차 |
|---:|---|---|---:|---|---:|---:|
{% for result in eda.education_gap_by_occupation[:5] %}
| {{ loop.index }} | {{ result.occupation }} | {{ result.lowest_education_group }} | {{ result.lowest_rate }}% | {{ result.highest_education_group }} | {{ result.highest_rate }}% | {{ result.gap_percentage_points }}%p |
{% endfor %}

이 격차는 관찰 데이터에서 나타난 연관성으로, 학력의 인과효과를 직접 의미하지는 않습니다.

![Top occupations by education gap](../charts/occupation_education_gap_top5.png)

## 6. 상관관계

![Correlation heatmap](../charts/correlation_heatmap.png)

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
