# src/analysis.py

# 화면이 없는 실행 환경에서도 Matplotlib 차트를 저장할 수 있도록 설정합니다.
import matplotlib

# Agg 백엔드는 GUI 창을 띄우지 않고 PNG 파일을 생성할 때 사용합니다.
# 반드시 pyplot을 import하기 전에 설정해야 합니다.
matplotlib.use("Agg")

# 정적 차트의 크기, 제목, 저장 등을 제어합니다.
import matplotlib.pyplot as plt

# 표 형태의 데이터 분석을 위해 사용합니다.
import pandas as pd

# 브라우저에서 확대, 축소, 마우스 확인이 가능한 인터랙티브 차트를 만듭니다.
import plotly.express as px

# Matplotlib을 기반으로 통계 시각화를 간단하게 작성할 수 있습니다.
import seaborn as sns

# 두 독립 집단의 평균 차이를 검정하기 위해 사용합니다.
from scipy.stats import ttest_ind

# 분석 결과 차트를 저장할 경로를 config.py에서 가져옵니다.
from src.config import (
    CORRELATION_CHART_PATH,
    EDUCATION_INCOME_CHART_PATH,
    EDUCATION_OCCUPATION_CHART_PATH,
    INCOME_CHART_PATH,
    INTERACTIVE_CHART_PATH,
)


EDUCATION_LABELS_EN = {
    "고졸 미만": "Below high school",
    "고졸": "High school",
    "대학 과정·전문학사": "College / Associate",
    "학사": "Bachelor's",
    "대학원 이상": "Graduate degree",
}


def add_analysis_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """학력 그룹과 고소득 여부 등 분석용 변수를 추가합니다."""

    analyzed = dataframe.copy()

    education_mapping = {
        "Preschool": "고졸 미만",
        "1st-4th": "고졸 미만",
        "5th-6th": "고졸 미만",
        "7th-8th": "고졸 미만",
        "9th": "고졸 미만",
        "10th": "고졸 미만",
        "11th": "고졸 미만",
        "12th": "고졸 미만",
        "HS-grad": "고졸",
        "Some-college": "대학 과정·전문학사",
        "Assoc-acdm": "대학 과정·전문학사",
        "Assoc-voc": "대학 과정·전문학사",
        "Bachelors": "학사",
        "Masters": "대학원 이상",
        "Prof-school": "대학원 이상",
        "Doctorate": "대학원 이상",
    }

    education_order = [
        "고졸 미만",
        "고졸",
        "대학 과정·전문학사",
        "학사",
        "대학원 이상",
    ]

    analyzed["education-group"] = (
        analyzed["education"].map(education_mapping)
    )

    analyzed["education-group"] = pd.Categorical(
        analyzed["education-group"],
        categories=education_order,
        ordered=True,
    )

    analyzed["high-income"] = (
        analyzed["income"] == ">50K"
    ).astype(int)

    return analyzed


def create_eda_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    """데이터의 기본 구조와 기술통계 결과를 사전 형태로 반환합니다."""

    # 숫자형 컬럼만 선택한 뒤 평균, 표준편차, 최솟값,
    # 사분위수, 최댓값 등의 기술통계를 계산합니다.
    numeric_summary = (
        dataframe.select_dtypes(include="number")
        .describe()

        # 결과를 소수점 둘째 자리까지 정리합니다.
        .round(2)

        # JSON 파일로 저장하기 쉽도록 DataFrame을 사전으로 변환합니다.
        .to_dict()
    )

    # income 컬럼에서 각 소득 등급이 몇 개인지 계산합니다.
    # 예: <=50K가 몇 행인지, >50K가 몇 행인지 확인합니다.
    income_counts = dataframe["income"].value_counts().to_dict()

    # 학력 그룹별 전체 인원과 고소득 비율을 계산합니다.
    analyzed = add_analysis_features(dataframe)
    education_income = (
        analyzed.groupby("education-group", observed=True)["high-income"]
        .agg(["count", "mean"])
    )

    education_income_summary = {
        str(group): {
            "count": int(row["count"]),
            "high_income_rate": round(float(row["mean"] * 100), 2),
        }
        for group, row in education_income.iterrows()
    }

    # EDA 결과를 하나의 사전으로 묶어 반환합니다.
    return {
        # 전체 행의 개수입니다.
        "rows": len(dataframe),

        # 전체 컬럼의 개수입니다.
        "columns": dataframe.shape[1],

        # 소득 등급별 데이터 개수를 저장합니다.
        "income_counts": {
            # JSON의 키와 값으로 안전하게 저장할 수 있도록
            # 키는 문자열, 값은 Python 정수로 변환합니다.
            str(key): int(value)
            for key, value in income_counts.items()
        },

        # 학력 그룹별 표본 수와 고소득 비율입니다.
        "education_income_summary": education_income_summary,

        # 숫자형 컬럼의 기술통계 결과입니다.
        "numeric_summary": numeric_summary,
    }


def create_visualizations(dataframe: pd.DataFrame) -> None:
    """소득, 상관관계, 학력·직업 및 근무시간 차트를 생성합니다."""

    # 차트 저장 폴더가 없으면 자동으로 생성합니다.
    INCOME_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 학력 그룹과 고소득 여부를 원본을 변경하지 않는 복사본에 추가합니다.
    analyzed = add_analysis_features(dataframe)

    # ---------------------------------------------------------
    # 1. 소득 등급별 데이터 개수 차트
    # ---------------------------------------------------------

    # 가로 7인치, 세로 5인치 크기의 새 그래프를 생성합니다.
    plt.figure(figsize=(7, 5))

    # income 컬럼의 등급별 데이터 개수를 막대그래프로 표현합니다.
    sns.countplot(
        data=dataframe,
        x="income",
        color="steelblue",
    )

    # 차트의 제목과 축 이름을 지정합니다.
    plt.title("Income Class Distribution")
    plt.xlabel("Income Class")
    plt.ylabel("Count")

    # 제목과 축 이름이 잘리지 않도록 여백을 자동 조정합니다.
    plt.tight_layout()

    # 차트를 PNG 파일로 저장합니다.
    # dpi=150은 이미지 해상도를 의미합니다.
    plt.savefig(
        INCOME_CHART_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    # 현재 그래프를 닫아 메모리를 정리합니다.
    # 여러 차트를 연속으로 생성할 때 반드시 필요합니다.
    plt.close()

    # ---------------------------------------------------------
    # 2. 수치형 변수 간 상관관계 히트맵
    # ---------------------------------------------------------

    # 상관관계를 확인할 주요 수치형 컬럼만 선택합니다.
    correlation_df = dataframe[
        [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
    ].copy()

    # income은 문자열 범주형 데이터이므로 상관계수를 계산할 수 없습니다.
    # <=50K는 0, >50K는 1로 변환한 새 컬럼을 추가합니다.
    correlation_df["income-binary"] = dataframe["income"].map(
        {
            "<=50K": 0,
            ">50K": 1,
        }
    )

    # 가로 9인치, 세로 7인치 크기의 새 그래프를 생성합니다.
    plt.figure(figsize=(9, 7))

    # 각 수치형 변수 사이의 상관계수를 히트맵으로 표현합니다.
    sns.heatmap(
        # corr()은 컬럼 간 피어슨 상관계수를 계산합니다.
        correlation_df.corr(),

        # 각 칸에 실제 상관계수 값을 표시합니다.
        annot=True,

        # 상관계수를 소수점 둘째 자리까지 표시합니다.
        fmt=".2f",

        # 음의 상관관계와 양의 상관관계를 색으로 구분합니다.
        cmap="coolwarm",

        # 상관계수 0을 색상 기준의 중앙값으로 설정합니다.
        center=0,
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    # 상관관계 히트맵을 PNG 파일로 저장합니다.
    plt.savefig(
        CORRELATION_CHART_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # 3. 학력 그룹별 고소득 비율 막대그래프
    # ---------------------------------------------------------

    education_income = (
        analyzed.groupby("education-group", observed=True)["high-income"]
        .agg(["count", "mean"])
        .reset_index()
    )
    education_income["high-income-rate"] = education_income["mean"] * 100
    education_income["education-label"] = education_income[
        "education-group"
    ].map(EDUCATION_LABELS_EN)

    plt.figure(figsize=(10, 6))
    axis = sns.barplot(
        data=education_income,
        x="education-label",
        y="high-income-rate",
        color="steelblue",
    )

    # 각 막대 위에 고소득 비율을 표시합니다.
    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f%%", padding=3)

    plt.title("High-income Rate by Education Level")
    plt.xlabel("Education Level")
    plt.ylabel("High-income Rate (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(
        EDUCATION_INCOME_CHART_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # ---------------------------------------------------------
    # 4. 학력 그룹과 직업별 고소득 비율 히트맵
    # ---------------------------------------------------------

    education_occupation = (
        analyzed.dropna(subset=["occupation"])
        .groupby(
            ["education-group", "occupation"],
            observed=True,
        )["high-income"]
        .agg(["count", "mean"])
        .reset_index()
    )

    # 표본이 너무 적은 조합은 비율이 불안정하므로 제외합니다.
    education_occupation = education_occupation.loc[
        education_occupation["count"] >= 30
    ].copy()
    education_occupation["high-income-rate"] = (
        education_occupation["mean"] * 100
    )

    occupation_heatmap = education_occupation.pivot(
        index="education-group",
        columns="occupation",
        values="high-income-rate",
    ).rename(index=EDUCATION_LABELS_EN)

    plt.figure(figsize=(17, 7))
    sns.heatmap(
        occupation_heatmap,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        linewidths=0.3,
        cbar_kws={"label": "High-income Rate (%)"},
    )
    plt.title("High-income Rate by Education Level and Occupation")
    plt.xlabel("Occupation")
    plt.ylabel("Education Level")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        EDUCATION_OCCUPATION_CHART_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # ---------------------------------------------------------
    # 5. 소득 집단별 주당 근무시간 박스 플롯
    # ---------------------------------------------------------

    # Plotly를 이용해 인터랙티브 박스 플롯을 생성합니다.
    figure = px.box(
        dataframe,

        # x축에는 소득 등급을 배치합니다.
        x="income",

        # y축에는 주당 근무시간을 배치합니다.
        y="hours-per-week",

        # 개별 데이터 점은 표시하지 않고 박스 플롯만 표시합니다.
        points=False,

        # 차트 제목을 지정합니다.
        title="Hours per Week by Income Class",

        # 원본 컬럼명을 읽기 쉬운 축 이름으로 변경합니다.
        labels={
            "income": "Income Class",
            "hours-per-week": "Hours per Week",
        },
    )

    # Plotly 차트를 독립적으로 실행할 수 있는 HTML 파일로 저장합니다.
    figure.write_html(
        INTERACTIVE_CHART_PATH,

        # Plotly JavaScript 파일을 HTML 안에 포함합니다.
        # 인터넷 연결이 없어도 차트를 열 수 있습니다.
        include_plotlyjs=True,

        # 완전한 HTML 문서 형태로 저장합니다.
        full_html=True,
    )


def run_statistics(dataframe: pd.DataFrame) -> dict[str, object]:
    """소득 집단별 교육 수준 평균 차이를 통계적으로 검정합니다."""

    # 소득이 50K 이하인 집단의 교육 수준 수치만 선택합니다.
    # dropna()는 결측치를 제거합니다.
    low_income = dataframe.loc[
        dataframe["income"] == "<=50K",
        "education-num",
    ].dropna()

    # 소득이 50K를 초과하는 집단의 교육 수준 수치만 선택합니다.
    high_income = dataframe.loc[
        dataframe["income"] == ">50K",
        "education-num",
    ].dropna()

    # 두 독립 집단의 평균 차이를 확인하기 위해 Welch t-test를 수행합니다.
    result = ttest_ind(
        low_income,
        high_income,

        # 두 집단의 분산이 같다고 가정하지 않습니다.
        # 이것이 일반적인 독립표본 t-test와 Welch t-test의 주요 차이입니다.
        equal_var=False,

        # 계산 과정에서 결측치가 있으면 제외합니다.
        nan_policy="omit",
    )

    # SciPy의 결과값을 일반 Python 실수형으로 변환합니다.
    p_value = float(result.pvalue)

    # p-value가 지나치게 작으면 컴퓨터에서는 0.0으로 표시될 수 있습니다.
    # 이 경우 실제로 0이라는 뜻이 아니라 1e-300보다 매우 작다는 의미입니다.
    p_value_display = (
        "< 1e-300"
        if p_value == 0.0
        else f"{p_value:.6e}"
    )

    # 유의수준 0.05를 기준으로 통계적 유의성을 해석합니다.
    if p_value < 0.05:
        interpretation = (
            "p-value가 0.05보다 작으므로 두 소득 집단의 교육 수준 "
            "평균은 통계적으로 유의한 차이가 있다고 해석합니다."
        )
    else:
        interpretation = (
            "p-value가 0.05 이상이므로 두 소득 집단의 교육 수준 "
            "평균 차이가 통계적으로 유의하다고 보기 어렵습니다."
        )

    # 전체 숫자형 컬럼 사이의 상관계수를 계산합니다.
    correlation = (
        dataframe.select_dtypes(include="number")
        .corr()
        .round(4)
        .to_dict()
    )

    # 통계 검정 결과와 상관계수를 하나의 사전으로 반환합니다.
    return {
        # 사용한 통계 검정의 이름입니다.
        "test": "Welch independent two-sample t-test",

        # 두 집단에서 비교한 변수입니다.
        "variable": "education-num",

        # 비교 대상 집단의 이름입니다.
        "group_low_income": "<=50K",
        "group_high_income": ">50K",

        # 각 집단에서 통계 검정에 사용한 데이터 개수입니다.
        "low_income_count": int(low_income.size),
        "high_income_count": int(high_income.size),

        # 각 집단의 평균 교육 수준 수치입니다.
        "low_income_mean": round(
            float(low_income.mean()),
            4,
        ),
        "high_income_mean": round(
            float(high_income.mean()),
            4,
        ),

        # t통계량은 두 집단 평균 차이의 상대적 크기를 나타냅니다.
        "t_statistic": round(
            float(result.statistic),
            6,
        ),

        # 계산에 사용하거나 후속 분석에 활용할 실제 p-value입니다.
        "p_value": p_value,

        # 사람이 읽기 쉬운 문자열 형태의 p-value입니다.
        "p_value_display": p_value_display,

        # 통계적 유의성 판단 기준입니다.
        "alpha": 0.05,

        # p-value에 따른 해석 문장입니다.
        "interpretation": interpretation,

        # 숫자형 변수 간 상관계수 결과입니다.
        "numeric_correlation": correlation,
    }
