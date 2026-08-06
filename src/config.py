# src/config.py

# 파일과 폴더 경로를 운영체제에 맞게 안전하게 다루기 위한 클래스입니다.
from pathlib import Path


# 현재 파일(config.py)의 위치를 기준으로 프로젝트 최상위 폴더를 찾습니다.
# __file__           : 현재 파일의 경로
# resolve()          : 절대 경로로 변환
# parent.parent      : src 폴더의 한 단계 위인 프로젝트 루트 폴더
BASE_DIR = Path(__file__).resolve().parent.parent


# Adult Census Income 원본 데이터를 내려받을 인터넷 주소입니다.
# 문자열을 여러 줄로 나누어 작성해도 Python에서는 하나의 문자열로 연결됩니다.
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/adult/adult.data"
)


# 원본 데이터 파일을 저장할 경로입니다.
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "adult.data"

# 전처리가 끝난 데이터를 CSV 형식으로 저장할 경로입니다.
PROCESSED_CSV_PATH = BASE_DIR / "data" / "processed" / "adult_cleaned.csv"

# 전처리가 끝난 데이터를 Parquet 형식으로 저장할 경로입니다.
# Parquet은 데이터 타입을 유지하고 대용량 데이터를 효율적으로 저장할 수 있습니다.
PROCESSED_PARQUET_PATH = (
    BASE_DIR / "data" / "processed" / "adult_cleaned.parquet"
)


# 분석 과정에서 생성되는 차트 이미지를 저장할 폴더입니다.
CHART_DIR = BASE_DIR / "outputs" / "charts"

# 데이터 분석 결과와 모델 평가 지표를 저장할 폴더입니다.
METRIC_DIR = BASE_DIR / "outputs" / "metrics"

# 학습이 끝난 머신러닝 모델을 저장할 폴더입니다.
MODEL_DIR = BASE_DIR / "outputs" / "models"

# 자동 생성한 Markdown 보고서를 저장할 폴더입니다.
REPORT_DIR = BASE_DIR / "outputs" / "reports"

# 보고서를 만들 때 사용할 Markdown 템플릿 파일의 경로입니다.
TEMPLATE_PATH = BASE_DIR / "templates" / "report_template.md"


# Pandas와 Polars의 처리 결과 및 성능 비교 정보를 저장할 파일입니다.
COMPARISON_PATH = METRIC_DIR / "pandas_polars_comparison.json"

# 탐색적 데이터 분석 결과를 저장할 파일입니다.
EDA_PATH = METRIC_DIR / "eda_summary.json"

# t-test 등 통계 분석 결과를 저장할 파일입니다.
STATISTICS_PATH = METRIC_DIR / "statistics.json"

# 정확도, 정밀도, 재현율, F1 점수 등 모델 평가 결과를 저장할 파일입니다.
MODEL_METRICS_PATH = METRIC_DIR / "model_metrics.json"

# 학습된 전처리 및 머신러닝 Pipeline을 저장할 파일입니다.
MODEL_PATH = MODEL_DIR / "income_pipeline.joblib"

# 최종 분석 보고서를 저장할 파일입니다.
REPORT_PATH = REPORT_DIR / "report.md"


# 소득 구간별 데이터 개수를 표현한 차트의 저장 경로입니다.
INCOME_CHART_PATH = CHART_DIR / "income_distribution.png"

# 수치형 변수 간 상관관계를 표현한 히트맵의 저장 경로입니다.
CORRELATION_CHART_PATH = CHART_DIR / "correlation_heatmap.png"

# Plotly로 생성한 인터랙티브 차트의 저장 경로입니다.
INTERACTIVE_CHART_PATH = CHART_DIR / "hours_by_income.html"

# 학력 그룹별 고소득 비율 막대그래프의 저장 경로입니다.
EDUCATION_INCOME_CHART_PATH = CHART_DIR / "education_income_rate.png"

# 학력 그룹과 직업별 고소득 비율 히트맵의 저장 경로입니다.
EDUCATION_OCCUPATION_CHART_PATH = (
    CHART_DIR / "education_occupation_income_heatmap.png"
)

# 같은 학력 안에서 최저·최고 직업 고소득률을 비교하는 차트 경로입니다.
EDUCATION_OCCUPATION_GAP_CHART_PATH = (
    CHART_DIR / "education_occupation_rate_gap.png"
)

# 학력 그룹별 고소득률 격차가 큰 직업 TOP 5 차트 경로입니다.
OCCUPATION_EDUCATION_GAP_CHART_PATH = (
    CHART_DIR / "occupation_education_gap_top5.png"
)

# 머신러닝 모델의 혼동행렬 차트를 저장할 경로입니다.
CONFUSION_MATRIX_PATH = CHART_DIR / "confusion_matrix.png"

# 모델 평가 지표를 막대그래프로 저장할 경로입니다.
MODEL_METRICS_CHART_PATH = CHART_DIR / "model_metrics.png"


# Adult 데이터 파일에는 헤더가 없으므로,
# 데이터의 각 열에 사용할 컬럼 이름을 직접 정의합니다.
COLUMN_NAMES = [
    "age",               # 나이
    "workclass",         # 고용 형태
    "fnlwgt",            # 인구 조사 가중치
    "education",         # 최종 학력
    "education-num",     # 학력 수준을 숫자로 표현한 값
    "marital-status",    # 결혼 상태
    "occupation",        # 직업
    "relationship",      # 가족 관계
    "race",              # 인종
    "sex",               # 성별
    "capital-gain",      # 자본 이익
    "capital-loss",      # 자본 손실
    "hours-per-week",    # 주당 근무 시간
    "native-country",    # 출신 국가
    "income",            # 예측 대상: 연 소득 50K 초과 여부
]


# 난수를 사용하는 작업의 결과를 항상 동일하게 만들기 위한 값입니다.
# 데이터 분할이나 모델 학습에서 재현성을 확보할 때 사용합니다.
RANDOM_STATE = 42

# 전체 데이터 중 20%를 테스트 데이터로 사용합니다.
# 나머지 80%는 모델 학습에 사용됩니다.
TEST_SIZE = 0.2
