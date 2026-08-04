# src/modeling.py

# 파일과 폴더 경로를 운영체제에 맞게 다루기 위해 사용합니다.
from pathlib import Path

# 학습이 끝난 머신러닝 모델을 파일로 저장하고 다시 불러오기 위해 사용합니다.
import joblib

# 화면이 없는 환경에서도 차트 이미지를 저장할 수 있도록 설정합니다.
import matplotlib

# GUI 창을 띄우지 않고 PNG 파일을 생성하는 백엔드입니다.
# pyplot을 불러오기 전에 설정해야 합니다.
matplotlib.use("Agg")

# 혼동행렬과 평가 지표 차트를 생성하고 저장하기 위해 사용합니다.
import matplotlib.pyplot as plt

# 저장 전후 예측 결과가 같은지 배열 단위로 비교하기 위해 사용합니다.
import numpy as np

# 입력 데이터를 DataFrame 형태로 처리하기 위해 사용합니다.
import pandas as pd

# 수치형 컬럼과 범주형 컬럼에 서로 다른 전처리를 적용합니다.
from sklearn.compose import ColumnTransformer

# 결측치를 중앙값이나 최빈값으로 채우기 위해 사용합니다.
from sklearn.impute import SimpleImputer

# 소득이 50K를 초과하는지 예측할 분류 모델입니다.
from sklearn.linear_model import LogisticRegression

# 머신러닝 모델의 성능을 측정하는 평가 함수입니다.
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# 전체 데이터를 학습용 데이터와 테스트용 데이터로 나눕니다.
from sklearn.model_selection import train_test_split

# 전처리와 모델 학습 단계를 하나의 작업 흐름으로 연결합니다.
from sklearn.pipeline import Pipeline

# 범주형 데이터 인코딩과 수치형 데이터 표준화를 위해 사용합니다.
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 차트 저장 경로, 난수값, 테스트 데이터 비율을 가져옵니다.
from src.config import (
    CONFUSION_MATRIX_PATH,
    MODEL_METRICS_CHART_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)


def train_evaluate_save_model(
    dataframe: pd.DataFrame,
    model_path: Path,
) -> dict[str, object]:
    """데이터를 학습하고 평가한 뒤 모델과 결과를 저장합니다."""

    # income 컬럼을 제외한 나머지 컬럼을 입력 데이터로 사용합니다.
    # 머신러닝에서 입력 데이터는 일반적으로 X라고 표현합니다.
    x_data = dataframe.drop(columns=["income"])

    # 문자열로 된 소득 등급을 머신러닝이 처리할 수 있는 숫자로 변환합니다.
    # <=50K는 0, >50K는 1로 표현합니다.
    # 예측하려는 정답 데이터는 일반적으로 y라고 표현합니다.
    y_data = dataframe["income"].map(
        {
            "<=50K": 0,
            ">50K": 1,
        }
    )

    # income 컬럼에 지정한 두 값 이외의 값이 있으면 NaN으로 변환됩니다.
    # 잘못된 정답 데이터로 모델을 학습하지 않도록 실행을 중단합니다.
    if y_data.isna().any():
        raise ValueError(
            "income 컬럼에 예상하지 못한 값이 있습니다."
        )

    # 입력 데이터에서 정수와 실수 등 숫자형 컬럼 이름을 찾습니다.
    numeric_features = (
        x_data.select_dtypes(include="number")
        .columns
        .tolist()
    )

    # 입력 데이터에서 문자열과 범주형 컬럼 이름을 찾습니다.
    categorical_features = (
        x_data.select_dtypes(
            include=["object", "string", "category"]
        )
        .columns
        .tolist()
    )

    # ---------------------------------------------------------
    # 1. 수치형 데이터 전처리 Pipeline
    # ---------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",

                # 수치형 결측치를 해당 컬럼의 중앙값으로 채웁니다.
                # 중앙값은 극단값의 영향을 평균보다 적게 받습니다.
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",

                # 수치형 컬럼의 크기 차이를 줄이기 위해 표준화합니다.
                # 평균은 0, 표준편차는 1에 가까운 값으로 변환합니다.
                StandardScaler(),
            ),
        ]
    )

    # ---------------------------------------------------------
    # 2. 범주형 데이터 전처리 Pipeline
    # ---------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",

                # 범주형 결측치를 가장 자주 나타나는 값으로 채웁니다.
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",

                # 문자로 된 범주를 0과 1로 구성된 숫자 컬럼으로 변환합니다.
                OneHotEncoder(
                    # 테스트 데이터에서 처음 보는 범주가 나타나도
                    # 오류를 발생시키지 않고 무시합니다.
                    handle_unknown="ignore",

                    # 변환 결과를 메모리 효율이 높은 희소 행렬로 만듭니다.
                    sparse_output=True,
                ),
            ),
        ]
    )

    # ---------------------------------------------------------
    # 3. 컬럼 종류에 따라 서로 다른 전처리 적용
    # ---------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            # 숫자형 컬럼에는 결측치 처리와 표준화를 적용합니다.
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),

            # 범주형 컬럼에는 결측치 처리와 원-핫 인코딩을 적용합니다.
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    # ---------------------------------------------------------
    # 4. 전처리와 머신러닝 모델을 하나의 Pipeline으로 연결
    # ---------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            # 먼저 수치형 및 범주형 데이터를 전처리합니다.
            ("preprocessor", preprocessor),

            (
                "model",

                # 전처리된 데이터를 이용해 이진 분류 모델을 학습합니다.
                LogisticRegression(
                    # 반복 횟수를 충분히 설정하여 수렴 가능성을 높입니다.
                    max_iter=1_000,

                    # 소규모 이진 분류에 적합한 최적화 알고리즘입니다.
                    solver="liblinear",

                    # 같은 데이터를 실행할 때 동일한 결과가 나오도록 설정합니다.
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # ---------------------------------------------------------
    # 5. 학습 데이터와 테스트 데이터 분리
    # ---------------------------------------------------------

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,

        # 전체 데이터 중 TEST_SIZE 비율을 테스트용으로 사용합니다.
        test_size=TEST_SIZE,

        # 실행할 때마다 동일한 데이터가 분리되도록 설정합니다.
        random_state=RANDOM_STATE,

        # 학습 데이터와 테스트 데이터의 소득 등급 비율을
        # 원본 데이터와 비슷하게 유지합니다.
        stratify=y_data,
    )

    # Pipeline 내부의 전처리와 모델 학습을 순서대로 실행합니다.
    # 전처리는 학습 데이터만 기준으로 학습되므로 데이터 누수를 방지합니다.
    pipeline.fit(x_train, y_train)

    # 학습에 사용하지 않은 테스트 데이터의 소득 등급을 예측합니다.
    predictions = pipeline.predict(x_test)

    # ---------------------------------------------------------
    # 6. 학습된 모델 저장과 재로딩 검증
    # ---------------------------------------------------------

    # 모델 파일을 저장할 상위 폴더가 없으면 생성합니다.
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 전처리 과정과 분류 모델이 포함된 전체 Pipeline을 저장합니다.
    joblib.dump(pipeline, model_path)

    # 방금 저장한 모델을 다시 불러옵니다.
    # 신뢰할 수 없는 joblib 파일은 보안상 불러오지 않아야 합니다.
    loaded_pipeline = joblib.load(model_path)

    # 다시 불러온 모델로 같은 테스트 데이터를 예측합니다.
    reloaded_predictions = loaded_pipeline.predict(x_test)

    # 저장 전 모델과 저장 후 다시 불러온 모델의 예측 결과가
    # 완전히 같은지 확인합니다.
    reloaded_match = bool(
        np.array_equal(
            predictions,
            reloaded_predictions,
        )
    )

    # ---------------------------------------------------------
    # 7. 혼동행렬 생성
    # ---------------------------------------------------------

    # 실제 정답과 예측값을 비교하여 혼동행렬을 계산합니다.
    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    # 혼동행렬 이미지를 저장할 폴더가 없으면 생성합니다.
    CONFUSION_MATRIX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 숫자 0과 1 대신 실제 소득 등급 이름을 표시합니다.
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["<=50K", ">50K"],
    )

    # 혼동행렬을 그래프로 그립니다.
    display.plot()

    plt.title("Confusion Matrix")
    plt.tight_layout()

    # 혼동행렬을 PNG 이미지로 저장합니다.
    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    # 현재 그래프를 닫아 메모리를 정리합니다.
    plt.close()

    # ---------------------------------------------------------
    # 8. 모델 평가 지표 계산
    # ---------------------------------------------------------

    scores = {
        # 전체 예측 중 올바르게 예측한 비율입니다.
        "Accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        # >50K라고 예측한 데이터 중 실제로 >50K인 비율입니다.
        "Precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        # 실제 >50K인 데이터 중 모델이 >50K로 찾아낸 비율입니다.
        "Recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        # Precision과 Recall의 균형을 나타내는 점수입니다.
        "F1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
    }

    # ---------------------------------------------------------
    # 9. 평가 지표 막대그래프 생성
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 5))

    # 평가 지표 이름을 x축, 점수를 y축으로 표현합니다.
    plt.bar(
        scores.keys(),
        scores.values(),
    )

    # 평가 지표의 범위가 0부터 1이므로 y축 범위를 고정합니다.
    plt.ylim(0, 1)

    plt.title("Model Evaluation Metrics")
    plt.ylabel("Score")
    plt.tight_layout()

    # 평가 지표 차트를 PNG 이미지로 저장합니다.
    plt.savefig(
        MODEL_METRICS_CHART_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # 10. 클래스별 상세 평가 결과 생성
    # ---------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,

        # 숫자 클래스 대신 실제 소득 등급 이름을 사용합니다.
        target_names=["<=50K", ">50K"],

        # 문자열이 아닌 사전 형태로 결과를 반환합니다.
        # 이후 JSON 파일과 Markdown 보고서에 저장하기 쉽습니다.
        output_dict=True,

        # 0으로 나누는 상황이 발생하면 점수를 0으로 처리합니다.
        zero_division=0,
    )

    # 모델 학습과 평가 결과를 하나의 사전으로 정리하여 반환합니다.
    return {
        # 사용한 머신러닝 모델의 이름입니다.
        "model": "LogisticRegression",

        # 결과 재현에 사용한 난수값입니다.
        "random_state": RANDOM_STATE,

        # 학습 데이터와 테스트 데이터의 행 개수입니다.
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),

        # 모델 학습에 사용한 수치형 및 범주형 컬럼입니다.
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,

        # 주요 평가 지표를 소수점 여섯째 자리까지 저장합니다.
        "accuracy": round(scores["Accuracy"], 6),
        "precision": round(scores["Precision"], 6),
        "recall": round(scores["Recall"], 6),
        "f1": round(scores["F1"], 6),

        # 혼동행렬을 JSON에 저장할 수 있도록 리스트로 변환합니다.
        "confusion_matrix": matrix.tolist(),

        # 클래스별 상세 평가 결과입니다.
        "classification_report": report,

        # 저장한 머신러닝 모델 파일의 경로입니다.
        "model_path": str(model_path),

        # 모델 저장 전후의 예측 결과가 같은지 나타냅니다.
        "reloaded_predictions_match": reloaded_match,
    }
