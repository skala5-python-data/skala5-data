# 학력과 직업에 따른 고소득 가능성 분석 및 예측

Adult Census Income 데이터를 활용하여 학력과 직업에 따른 고소득 비율을 분석하고, 머신러닝 모델로 연 소득 50K 초과 여부를 예측하는 End-to-End 데이터 분석 프로젝트입니다.

데이터 다운로드부터 정제, EDA, 시각화, 통계 검정, 머신러닝 모델 학습, 보고서 생성까지 한 번의 명령으로 실행할 수 있습니다.

---

## 1. 주요 기능

이 프로젝트는 다음 작업을 자동으로 수행합니다.

1. Adult Census Income 원본 데이터 다운로드
2. Pandas와 Polars를 이용한 데이터 로딩 및 성능 비교
3. 중복값과 결측치 처리
4. 탐색적 데이터 분석 및 기술통계 계산
5. Seaborn·Matplotlib 정적 차트 생성
6. Plotly 인터랙티브 차트 생성
7. 학력 및 직업별 고소득률 분석
8. 상관계수 계산 및 Welch t-test
9. sklearn Pipeline 기반 머신러닝 모델 학습
10. 정확도·정밀도·재현율·F1 평가
11. joblib 모델 저장 및 재로딩 검증
12. Markdown 보고서 자동 생성

---

## 2. 실행 환경

### 권장 환경

- Python 3.11 이상
- Python 3.12 권장
- macOS, Linux 또는 Windows
- 첫 실행 시 데이터 다운로드를 위한 인터넷 연결

Python 버전은 다음 명령으로 확인할 수 있습니다.

```
python --version
```

macOS에서 `python` 명령이 없다면 다음 명령을 사용합니다.

```
python3 --version
```

---

## 3. 프로젝트 폴더 구조

```
skala5-data/
├── data/
│   ├── raw/
│   │   └── adult.data
│   └── processed/
│       ├── adult_cleaned.csv
│       └── adult_cleaned.parquet
├── outputs/
│   ├── charts/
│   ├── metrics/
│   ├── models/
│   └── reports/
├── src/
│   ├── analysis.py
│   ├── config.py
│   ├── data_pipeline.py
│   ├── io_utils.py
│   ├── main.py
│   ├── modeling.py
│   └── reporting.py
├── templates/
│   └── report_template.md
├── requirements.txt
└── README.md
```

각 파일의 역할은 다음과 같습니다.

| 파일 | 역할 |
| --- | --- |
| `src/main.py` | 전체 분석 과정의 실행 순서 관리 |
| `src/config.py` | 데이터, 모델, 차트, 보고서 경로 및 설정값 관리 |
| `src/data_pipeline.py` | 데이터 다운로드, Pandas·Polars 로딩 및 정제 |
| `src/analysis.py` | EDA, 시각화, 상관분석 및 통계 검정 |
| `src/modeling.py` | 머신러닝 Pipeline 학습, 평가 및 모델 저장 |
| `src/reporting.py` | Jinja2 기반 Markdown 보고서 생성 |
| `src/io_utils.py` | 분석 결과를 JSON 파일로 저장 |
| `templates/report_template.md` | 자동 생성 보고서의 템플릿 |
| `requirements.txt` | 실행에 필요한 Python 라이브러리 목록 |

---

## 4. 프로젝트 다운로드

Git 저장소를 복제한 뒤 프로젝트 폴더로 이동합니다.

```
git clone <저장소-URL>
cd skala5-data
```

이미 프로젝트 파일을 내려받았다면 터미널에서 해당 폴더로 이동합니다.

```
cd skala5-data
```

명령어는 반드시 `src`, `data`, `outputs` 폴더가 보이는 프로젝트 최상위 폴더에서 실행해야 합니다.

---

## 5. 가상환경 생성

프로젝트별로 라이브러리를 분리해서 관리하기 위해 가상환경 사용을 권장합니다.

### macOS 또는 Linux

가상환경을 생성합니다.

```
python3 -m venv .venv
```

가상환경을 활성화합니다.

```
source .venv/bin/activate
```

활성화되면 터미널 앞에 다음과 같이 `(.venv)`가 표시됩니다.

```
(.venv) ➜ skala5-data
```

### Windows PowerShell

가상환경을 생성합니다.

```
py -m venv .venv
```

가상환경을 활성화합니다.

```
.venv\Scripts\Activate.ps1
```

PowerShell 실행 정책 때문에 활성화가 차단되면 다음 명령을 실행한 뒤 다시 시도합니다.

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

---

## 6. 라이브러리 설치

가상환경을 활성화한 상태에서 pip를 업데이트합니다.

```
python -m pip install --upgrade pip
```

그다음 필요한 라이브러리를 설치합니다.

```
python -m pip install -r requirements.txt
```

설치되는 주요 라이브러리는 다음과 같습니다.

| 라이브러리 | 용도 |
| --- | --- |
| `numpy` | 수치 계산 및 배열 처리 |
| `pandas` | 데이터 로딩, 정제 및 분석 |
| `polars` | 데이터 처리 성능 비교 |
| `pyarrow` | Parquet 파일 저장 및 로딩 |
| `scipy` | Welch t-test 수행 |
| `scikit-learn` | 전처리 Pipeline 및 머신러닝 모델 |
| `matplotlib` | 정적 차트 생성 |
| `seaborn` | 통계 시각화 |
| `plotly` | 인터랙티브 차트 생성 |
| `joblib` | 학습된 모델 저장 및 로딩 |
| `jinja2` | Markdown 보고서 자동 생성 |

설치 여부를 확인하려면 다음 명령을 사용할 수 있습니다.

```
python -m pip list
```

---

## 7. 프로젝트 실행

프로젝트 최상위 폴더에서 다음 명령을 실행합니다.

```
python -m src.main
```

macOS에서 가상환경을 사용하지 않고 `python` 명령을 찾을 수 없다면 다음 명령을 사용합니다.

```
python3 -m src.main
```

`python src/main.py`가 아닌 `python -m src.main`으로 실행하는 것을 권장합니다. 프로젝트 내부에서 `src` 패키지를 기준으로 모듈을 불러오기 때문입니다.

---

## 8. 실행 과정

프로그램을 실행하면 다음 순서로 진행됩니다.

```
[1/8] 데이터 파일을 준비합니다.
[2/8] Pandas와 Polars로 데이터를 읽습니다.
[3/8] 결측치와 중복 데이터를 정리합니다.
[4/8] 탐색적 데이터 분석 결과를 생성합니다.
[5/8] 정적 차트와 인터랙티브 차트를 생성합니다.
[6/8] 교육 수준 차이와 상관관계를 분석합니다.
[7/8] 머신러닝 Pipeline을 학습하고 저장합니다.
[8/8] Markdown 보고서를 자동 생성합니다.
```

모든 작업이 완료되면 다음과 같은 결과가 출력됩니다.

```
전체 분석이 완료되었습니다.
정제 후 데이터: 32537행 × 15열
정확도: 0.85756
F1 점수: 0.679571
저장 모델 재로딩 일치: True
```

실행 시간과 모델 성능은 Python 버전, 운영체제 및 라이브러리 버전에 따라 조금 달라질 수 있습니다.

---

## 9. 데이터 다운로드

첫 실행 시 원본 Adult 데이터가 없다면 자동으로 다운로드합니다.

다운로드 위치:

```
data/raw/adult.data
```

이미 정상적인 파일이 있으면 다시 다운로드하지 않습니다.

인터넷 연결 문제로 자동 다운로드가 실패하면 Adult 데이터 파일을 직접 내려받아 다음 위치에 저장합니다.

```
data/raw/adult.data
```

파일 이름이 정확히 `adult.data`인지 확인해야 합니다.

---

## 10. 데이터 정제 방식

다음 기준으로 데이터를 정제합니다.

- 문자열 앞뒤 공백 제거
- `?`를 결측치로 변환
- 완전히 동일한 중복 행 제거
- 범주형 결측치를 `Unknown` 범주로 대체
- Pandas와 Polars에 동일한 정제 기준 적용
- 정제 후 두 결과의 행과 열 크기 비교

머신러닝 Pipeline에는 새로운 데이터의 결측치에 대비해 다음 전처리를 포함합니다.

- 수치형 결측치: 중앙값 대체
- 범주형 결측치: 최빈값 대체
- 수치형 변수: 표준화
- 범주형 변수: 원-핫 인코딩

---

## 11. 주요 분석 내용

프로젝트에서는 다음 질문을 분석합니다.

1. 학력 수준이 높을수록 고소득 비율이 증가하는가?
2. 같은 학력에서도 직업별 고소득률이 달라지는가?
3. 학력 그룹 간 고소득률 격차가 큰 직업은 무엇인가?
4. 두 소득 집단의 평균 교육 수준에 차이가 있는가?
5. 개인 특성으로 연 소득 50K 초과 여부를 예측할 수 있는가?

직업별 분석에서는 작은 표본으로 인해 비율이 과장되는 것을 방지하기 위해 학력·직업 조합별 표본이 30명 이상인 경우만 비교합니다.

---

## 12. 머신러닝 모델

머신러닝 모델은 sklearn의 `Pipeline`을 이용해 구성합니다.

```
수치형 데이터
  → 중앙값 대체
  → StandardScaler

범주형 데이터
  → 최빈값 대체
  → OneHotEncoder

전처리 결과
  → LogisticRegression
```

전체 데이터 중 80%를 학습 데이터로, 20%를 테스트 데이터로 사용합니다.

클래스 비율을 유지하기 위해 `stratify`를 적용하고, 동일한 결과를 재현할 수 있도록 `random_state=42`를 사용합니다.

평가 지표:

- 정확도
- 정밀도
- 재현율
- F1 점수
- 혼동행렬

---

## 13. 주요 산출물

### 정제 데이터

```
data/processed/adult_cleaned.csv
data/processed/adult_cleaned.parquet
```

### 분석 결과

```
outputs/metrics/pandas_polars_comparison.json
outputs/metrics/eda_summary.json
outputs/metrics/statistics.json
outputs/metrics/model_metrics.json
```

### 정적 차트

```
outputs/charts/income_distribution.png
outputs/charts/correlation_heatmap.png
outputs/charts/education_income_rate.png
outputs/charts/education_occupation_income_heatmap.png
outputs/charts/education_occupation_rate_gap.png
outputs/charts/occupation_education_gap_top5.png
outputs/charts/model_metrics.png
outputs/charts/confusion_matrix.png
```

### Plotly 인터랙티브 차트

```
outputs/charts/hours_by_income.html
```

HTML 파일을 브라우저로 열면 확대, 축소 및 마우스 오버 기능을 사용할 수 있습니다.

### 머신러닝 모델

```
outputs/models/income_pipeline.joblib
```

### 자동 생성 보고서

```
outputs/reports/report.md
```

---

## 14. 결과 확인 방법

### Markdown 보고서 확인

다음 파일을 Markdown 미리보기로 엽니다.

```
outputs/reports/report.md
```

### 정적 차트 확인

다음 폴더의 PNG 파일을 엽니다.

```
outputs/charts/
```

### Plotly 차트 확인

다음 HTML 파일을 웹 브라우저로 엽니다.

```
outputs/charts/hours_by_income.html
```

### 모델 평가 결과 확인

다음 JSON 파일을 엽니다.

```
outputs/metrics/model_metrics.json
```

---

## 15. 자주 발생하는 문제

### `python: command not found`

macOS에서는 다음 명령을 사용합니다.

```
python3 -m src.main
```

가상환경을 사용한다면 먼저 활성화합니다.

```
source .venv/bin/activate
```

### `ModuleNotFoundError`

프로젝트 최상위 폴더에서 실행했는지 확인합니다.

```
pwd
ls
```

현재 위치에서 `src`, `data`, `outputs`, `requirements.txt`가 보여야 합니다.

라이브러리를 다시 설치합니다.

```
python -m pip install -r requirements.txt
```

### Parquet 저장 오류

`pyarrow`가 설치됐는지 확인합니다.

```
python -m pip install pyarrow
```

### 데이터 다운로드 오류

인터넷 연결을 확인하거나 원본 파일을 직접 다음 위치에 저장합니다.

```
data/raw/adult.data
```

### Matplotlib 캐시 경고

차트는 생성되지만 캐시 폴더 관련 경고가 발생한다면 쓰기 가능한 캐시 경로를 지정할 수 있습니다.

macOS 또는 Linux:

```
export MPLCONFIGDIR=/tmp/matplotlib-cache
python -m src.main
```

### Plotly HTML이 GitHub에서 보이지 않음

현재 `.gitignore` 설정에 따라 HTML 파일이 Git에 포함되지 않을 수 있습니다. 해당 파일은 다음 명령으로 프로젝트를 실행하면 다시 생성됩니다.

```
python -m src.main
```

---

## 16. 가상환경 종료

작업이 끝나면 다음 명령으로 가상환경을 종료합니다.

```
deactivate
```

---

## 17. 빠른 실행 요약

macOS 또는 Linux:

```
git clone <저장소-URL>
cd skala5-data
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.main
```

Windows PowerShell:

```
git clone <저장소-URL>
cd skala5-data
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.main
```