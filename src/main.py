# src/main.py

# 탐색적 데이터 분석, 시각화, 통계 분석에 필요한 함수를 가져옵니다.
from src.analysis import (
    add_analysis_features,
    create_eda_summary,
    create_visualizations,
    run_statistics,
)

# 데이터, 결과 파일, 모델, 보고서의 저장 경로를 가져옵니다.
from src.config import (
    COMPARISON_PATH,
    DATA_URL,
    EDA_PATH,
    MODEL_METRICS_PATH,
    MODEL_PATH,
    PROCESSED_CSV_PATH,
    PROCESSED_PARQUET_PATH,
    RAW_DATA_PATH,
    REPORT_PATH,
    STATISTICS_PATH,
    TEMPLATE_PATH,
)

# 데이터 다운로드, 로딩, 정제, 비교에 필요한 함수를 가져옵니다.
from src.data_pipeline import (
    build_comparison,
    clean_pandas,
    clean_polars,
    download_data,
    load_with_pandas,
    load_with_polars,
)

# 분석 결과를 JSON 파일로 저장하는 함수를 가져옵니다.
from src.io_utils import save_json

# 머신러닝 Pipeline을 학습하고 평가하는 함수를 가져옵니다.
from src.modeling import train_evaluate_save_model

# 분석 결과를 Markdown 보고서로 생성하는 함수를 가져옵니다.
from src.reporting import generate_report


def main() -> None:
    """데이터 준비부터 보고서 생성까지 전체 분석 과정을 순서대로 실행합니다."""

    # ---------------------------------------------------------
    # 1. 원본 데이터 준비
    # ---------------------------------------------------------

    print("[1/8] 데이터 파일을 준비합니다.")

    # 원본 데이터 파일이 없으면 인터넷에서 다운로드합니다.
    # 파일이 이미 존재하면 다시 다운로드하지 않습니다.
    download_data(
        DATA_URL,
        RAW_DATA_PATH,
    )

    # ---------------------------------------------------------
    # 2. Pandas와 Polars 데이터 로딩
    # ---------------------------------------------------------

    print("[2/8] Pandas와 Polars로 데이터를 읽습니다.")

    # 같은 데이터를 Pandas로 읽고 로딩 시간을 함께 받습니다.
    pandas_raw, pandas_seconds = load_with_pandas(
        RAW_DATA_PATH
    )

    # 같은 데이터를 Polars로 읽고 로딩 시간을 함께 받습니다.
    polars_raw, polars_seconds = load_with_polars(
        RAW_DATA_PATH
    )

    # ---------------------------------------------------------
    # 3. 결측치와 중복 데이터 정제
    # ---------------------------------------------------------

    print("[3/8] 결측치와 중복 데이터를 정리합니다.")

    # Pandas 데이터의 문자열 공백, 결측치, 중복값을 정리합니다.
    # pandas_cleaning에는 정제 전후의 행 수와 결측치 개수가 저장됩니다.
    pandas_clean, pandas_cleaning = clean_pandas(
        pandas_raw
    )

    # Polars 데이터에도 같은 정제 기준을 적용합니다.
    polars_clean, polars_cleaning = clean_polars(
        polars_raw
    )

    # Pandas와 Polars의 처리 결과를 비교합니다.
    # 최종 행과 열의 개수가 다르면 build_comparison()에서 오류가 발생합니다.
    comparison = build_comparison(
        pandas_clean,
        pandas_seconds,
        pandas_cleaning,
        polars_clean,
        polars_seconds,
        polars_cleaning,
    )

    # 로딩 시간, 메모리 사용량, 정제 결과를 JSON 파일로 저장합니다.
    save_json(
        comparison,
        COMPARISON_PATH,
    )

    # 전처리된 데이터를 저장할 폴더가 없으면 생성합니다.
    PROCESSED_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 정제된 데이터를 CSV 형식으로 저장합니다.
    # utf-8-sig는 Excel에서 한글이 깨지는 문제를 줄여 줍니다.
    pandas_clean.to_csv(
        PROCESSED_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # 정제된 데이터를 Parquet 형식으로도 저장합니다.
    # Parquet은 데이터 타입을 유지하며 저장 공간과 처리 속도에 유리합니다.
    pandas_clean.to_parquet(
        PROCESSED_PARQUET_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # 4. 탐색적 데이터 분석
    # ---------------------------------------------------------

    print("[4/8] 탐색적 데이터 분석 결과를 생성합니다.")

    # 전체 행과 열의 수, 소득 등급별 개수,
    # 숫자형 컬럼의 기술통계를 계산합니다.
    eda_summary = create_eda_summary(
        pandas_clean
    )

    # 학력 그룹별 전체 인원과 평균 고소득 여부를 집계합니다.
    analyzed = add_analysis_features(pandas_clean)
    education_income = (
        analyzed.groupby(
            "education-group",
            observed=True,
        )["high-income"]
        .agg(["count", "mean"])
    )
    education_income["high_income_rate"] = (
        education_income["mean"] * 100
    )

    # 별도의 Python 스크립트를 실행하지 않아도 메인 실행 결과에서
    # 학력 그룹별 고소득 비율을 바로 확인할 수 있도록 출력합니다.
    print("\n학력 그룹별 고소득 분석 결과")
    print(education_income)

    # 같은 학력 안에서 직업별 고소득률 차이를 출력합니다.
    print("\n같은 학력 내 직업별 고소득률 차이")
    for education_group, result in eda_summary[
        "occupation_gap_by_education"
    ].items():
        print(
            f"{education_group}: "
            f"{result['lowest_occupation']} {result['lowest_rate']}% → "
            f"{result['highest_occupation']} {result['highest_rate']}% "
            f"(격차 {result['gap_percentage_points']}%p)"
        )

    # 학력 그룹 간 고소득률 격차가 가장 큰 직업 5개를 출력합니다.
    print("\n학력에 따른 고소득률 격차가 큰 직업 TOP 5")
    for rank, result in enumerate(
        eda_summary["education_gap_by_occupation"][:5],
        start=1,
    ):
        print(
            f"{rank}. {result['occupation']}: "
            f"{result['lowest_education_group']} {result['lowest_rate']}% → "
            f"{result['highest_education_group']} {result['highest_rate']}% "
            f"(격차 {result['gap_percentage_points']}%p)"
        )

    # 탐색적 데이터 분석 결과를 JSON 파일로 저장합니다.
    save_json(
        eda_summary,
        EDA_PATH,
    )

    # ---------------------------------------------------------
    # 5. 데이터 시각화
    # ---------------------------------------------------------

    print("[5/8] 정적 차트와 인터랙티브 차트를 생성합니다.")

    # 소득 분포 차트, 상관관계 히트맵,
    # 소득 집단별 근무시간 박스 플롯을 생성합니다.
    create_visualizations(
        pandas_clean
    )

    # ---------------------------------------------------------
    # 6. 통계 분석
    # ---------------------------------------------------------

    print("[6/8] 교육 수준 차이와 상관관계를 분석합니다.")

    # 소득 집단별 교육 수준 차이를 Welch t-test로 검정하고,
    # 숫자형 변수 간 상관계수를 계산합니다.
    statistics = run_statistics(
        pandas_clean
    )

    # 통계 분석 결과를 JSON 파일로 저장합니다.
    save_json(
        statistics,
        STATISTICS_PATH,
    )

    # ---------------------------------------------------------
    # 7. 머신러닝 모델 학습과 평가
    # ---------------------------------------------------------

    print("[7/8] 머신러닝 Pipeline을 학습하고 저장합니다.")

    # 전처리와 로지스틱 회귀 모델을 하나의 Pipeline으로 학습합니다.
    # 정확도, 정밀도, 재현율, F1 점수도 함께 계산합니다.
    model_metrics = train_evaluate_save_model(
        pandas_clean,
        MODEL_PATH,
    )

    # 모델 평가 결과를 JSON 파일로 저장합니다.
    save_json(
        model_metrics,
        MODEL_METRICS_PATH,
    )

    # ---------------------------------------------------------
    # 8. Markdown 보고서 자동 생성
    # ---------------------------------------------------------

    print("[8/8] Markdown 보고서를 자동 생성합니다.")

    # 앞 단계에서 생성한 비교, EDA, 통계, 모델 평가 결과를
    # Jinja2 템플릿에 전달하여 최종 보고서를 만듭니다.
    generate_report(
        TEMPLATE_PATH,
        REPORT_PATH,
        comparison,
        eda_summary,
        statistics,
        model_metrics,
    )

    # ---------------------------------------------------------
    # 전체 실행 결과 출력
    # ---------------------------------------------------------

    print("\n전체 분석이 완료되었습니다.")

    # 정제 후 데이터의 크기를 출력합니다.
    print(
        f"정제 후 데이터: "
        f"{eda_summary['rows']}행 × "
        f"{eda_summary['columns']}열"
    )

    # 주요 머신러닝 평가 결과를 출력합니다.
    print(f"정확도: {model_metrics['accuracy']}")
    print(f"F1 점수: {model_metrics['f1']}")

    # 소득 집단별 교육 수준 차이 검정 결과를 출력합니다.
    print("\n소득 집단별 교육 수준 차이 검정 결과")
    print(f"분석 변수: {statistics['variable']}")
    print(
        f"<=50K 집단 평균: "
        f"{statistics['low_income_mean']}"
    )
    print(
        f">50K 집단 평균: "
        f"{statistics['high_income_mean']}"
    )
    print(f"t통계량: {statistics['t_statistic']}")
    print(f"p-value: {statistics['p_value_display']}")

    # 저장한 모델을 다시 불러온 뒤에도 예측 결과가 같은지 출력합니다.
    print(
        "저장 모델 재로딩 일치: "
        f"{model_metrics['reloaded_predictions_match']}"
    )

    # 최종 산출물이 저장된 경로를 출력합니다.
    print(f"정제 데이터: {PROCESSED_PARQUET_PATH}")
    print(f"모델 파일: {MODEL_PATH}")
    print(f"최종 보고서: {REPORT_PATH}")


# 이 파일을 직접 실행할 때만 main() 함수를 호출합니다.
# 다른 파일에서 import할 때는 자동으로 실행되지 않습니다.
if __name__ == "__main__":
    main()
