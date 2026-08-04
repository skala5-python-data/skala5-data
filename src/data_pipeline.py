# src/data_pipeline.py

# 파일과 폴더 경로를 운영체제에 맞게 다루기 위해 사용합니다.
from pathlib import Path

# 코드 실행 시간을 정밀하게 측정하기 위해 사용합니다.
from time import perf_counter

# 인터넷 연결 또는 URL 접근 오류를 처리하기 위해 사용합니다.
from urllib.error import URLError

# 지정한 URL의 파일을 로컬 경로로 내려받기 위해 사용합니다.
from urllib.request import urlretrieve

# Pandas는 표 형태의 데이터를 분석할 때 사용하는 라이브러리입니다.
import pandas as pd

# Polars는 빠른 데이터 처리에 초점을 둔 DataFrame 라이브러리입니다.
import polars as pl

# Adult 데이터에 사용할 컬럼 이름 목록을 가져옵니다.
from src.config import COLUMN_NAMES


def download_data(url: str, destination: Path) -> None:
    """데이터 파일이 없을 때만 인터넷에서 내려받습니다."""

    # 대상 파일이 이미 존재하고 크기가 0보다 크면 다시 다운로드하지 않습니다.
    # 기존 파일을 보호하고 불필요한 네트워크 요청을 줄이기 위한 처리입니다.
    if destination.exists() and destination.stat().st_size > 0:
        return

    # 데이터를 저장할 상위 폴더가 없으면 자동으로 생성합니다.
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        # URL의 파일을 destination 경로에 저장합니다.
        urlretrieve(url, destination)

    except URLError as error:
        # 인터넷 연결, 주소 접근, 서버 오류 등이 발생하면
        # 초보자가 해결 방법을 알 수 있도록 명확한 메시지를 제공합니다.
        raise RuntimeError(
            "데이터 다운로드에 실패했습니다. 배포본의 adult.data 파일을 "
            f"{destination} 경로에 복사한 뒤 다시 실행하세요."
        ) from error


def load_with_pandas(path: Path) -> tuple[pd.DataFrame, float]:
    """Adult 데이터를 Pandas로 읽고 로딩 시간을 반환합니다."""

    # 데이터 읽기를 시작하기 직전의 시간을 기록합니다.
    start = perf_counter()

    dataframe = pd.read_csv(
        path,

        # 원본 데이터에는 컬럼 제목이 없으므로 헤더가 없다고 지정합니다.
        header=None,

        # config.py에서 정의한 컬럼 이름을 적용합니다.
        names=COLUMN_NAMES,

        # 쉼표 뒤에 있는 불필요한 공백을 제거하면서 읽습니다.
        skipinitialspace=True,

        # 물음표(?)는 실제 값이 없는 결측치로 처리합니다.
        na_values=["?"],
    )

    # 데이터 읽기가 끝난 뒤 경과 시간을 계산합니다.
    elapsed = perf_counter() - start

    # 읽어온 DataFrame과 로딩 시간을 함께 반환합니다.
    return dataframe, elapsed


def load_with_polars(path: Path) -> tuple[pl.DataFrame, float]:
    """Adult 데이터를 Polars로 읽고 로딩 시간을 반환합니다."""

    # Pandas와 같은 기준으로 로딩 시간을 측정합니다.
    start = perf_counter()

    dataframe = pl.read_csv(
        path,

        # 원본 데이터에 컬럼 제목이 없음을 지정합니다.
        has_header=False,

        # 읽어온 데이터에 컬럼 이름을 적용합니다.
        new_columns=COLUMN_NAMES,

        # 각 컬럼은 쉼표로 구분되어 있습니다.
        separator=",",

        # 공백이 포함된 물음표와 일반 물음표를 모두 결측치로 처리합니다.
        null_values=["?", " ?"],

        # 앞부분 10,000행을 확인하여 각 컬럼의 자료형을 추론합니다.
        infer_schema_length=10_000,
    )

    # 파일 로딩에 걸린 시간만 측정합니다.
    elapsed = perf_counter() - start

    # 문자열 자료형을 가진 컬럼 이름만 추출합니다.
    string_columns = [
        name
        for name, dtype in dataframe.schema.items()
        if dtype == pl.String
    ]

    # 문자열 컬럼이 존재할 때만 공백과 물음표를 정리합니다.
    if string_columns:
        dataframe = dataframe.with_columns(
            [
                # 문자열 앞뒤 공백을 제거한 결과가 "?"이면 null로 바꿉니다.
                pl.when(pl.col(name).str.strip_chars() == "?")
                .then(None)

                # "?"가 아니라면 앞뒤 공백만 제거한 값을 사용합니다.
                .otherwise(pl.col(name).str.strip_chars())

                # 처리한 결과를 기존 컬럼 이름으로 다시 저장합니다.
                .alias(name)
                for name in string_columns
            ]
        )

    return dataframe, elapsed


def clean_pandas(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Pandas DataFrame의 문자열, 중복값, 결측치를 정리합니다."""

    # 원본 DataFrame을 직접 수정하지 않도록 복사본을 만듭니다.
    cleaned = dataframe.copy()

    # 문자열 자료형을 가진 컬럼만 선택합니다.
    text_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns

    # 모든 문자열 컬럼의 앞뒤 공백과 물음표 값을 정리합니다.
    for column in text_columns:
        cleaned[column] = cleaned[column].str.strip()
        cleaned[column] = cleaned[column].replace("?", pd.NA)

    # 정제 전 데이터 상태를 기록합니다.
    rows_before = len(cleaned)
    duplicates_before = int(cleaned.duplicated().sum())
    missing_before = int(cleaned.isna().sum().sum())

    # 완전히 동일한 행을 제거하고 인덱스를 0부터 다시 부여합니다.
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 예측 대상인 income 값이 없는 행은 모델 학습에 사용할 수 없으므로 제거합니다.
    cleaned = cleaned.dropna(subset=["income"]).reset_index(drop=True)

    # 데이터 정제 전후 상태를 사전 형태로 정리합니다.
    summary = {
        "rows_before": rows_before,
        "rows_after": len(cleaned),
        "duplicates_removed": duplicates_before,
        "missing_values_before": missing_before,
        "missing_values_after": int(cleaned.isna().sum().sum()),
    }

    return cleaned, summary


def clean_polars(
    dataframe: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Polars DataFrame의 중복값과 결측치를 정리합니다."""

    # Polars에서는 height가 행의 개수를 의미합니다.
    rows_before = dataframe.height

    # 전체 행 수에서 고유한 행 수를 빼서 중복 행 수를 계산합니다.
    duplicates_before = rows_before - dataframe.unique().height

    # 각 컬럼의 null 개수를 더하여 전체 결측치 개수를 계산합니다.
    missing_before = int(sum(dataframe.null_count().row(0)))

    # 중복 행을 제거합니다.
    # maintain_order=True는 원본 데이터의 행 순서를 유지합니다.
    cleaned = dataframe.unique(maintain_order=True).filter(
        # 예측 대상인 income 값이 null인 행은 제거합니다.
        pl.col("income").is_not_null()
    )

    # 정제 전후 결과를 사전 형태로 정리합니다.
    summary = {
        "rows_before": rows_before,
        "rows_after": cleaned.height,
        "duplicates_removed": duplicates_before,
        "missing_values_before": missing_before,
        "missing_values_after": int(
            sum(cleaned.null_count().row(0))
        ),
    }

    return cleaned, summary


def build_comparison(
    pandas_df: pd.DataFrame,
    pandas_seconds: float,
    pandas_cleaning: dict[str, int],
    polars_df: pl.DataFrame,
    polars_seconds: float,
    polars_cleaning: dict[str, int],
) -> dict[str, object]:
    """Pandas와 Polars의 처리 결과 및 성능 정보를 비교합니다."""

    # 두 라이브러리로 정제한 데이터의 행 수와 열 수가 같은지 확인합니다.
    # 결과 크기가 다르면 공백, 결측치, 중복 처리 방식이 다를 가능성이 있습니다.
    same_shape = (
        len(pandas_df) == polars_df.height
        and pandas_df.shape[1] == polars_df.width
    )

    # 동일한 원본 데이터를 처리했으므로 최종 크기도 같아야 합니다.
    # 다를 경우 잘못된 비교 결과를 저장하지 않고 즉시 실행을 중단합니다.
    if not same_shape:
        raise RuntimeError(
            "Pandas와 Polars의 정제 결과 크기가 다릅니다. "
            "공백과 결측치 처리 코드를 확인하세요."
        )

    # Pandas와 Polars의 데이터 크기, 로딩 시간, 메모리 사용량,
    # 정제 결과를 하나의 사전으로 구성하여 반환합니다.
    return {
        "pandas": {
            # Pandas의 행과 열 개수입니다.
            "rows": len(pandas_df),
            "columns": pandas_df.shape[1],

            # 소수점 이하 6자리까지 로딩 시간을 저장합니다.
            "load_seconds": round(pandas_seconds, 6),

            # 문자열 등 실제 데이터가 차지하는 메모리까지 계산합니다.
            "memory_bytes": int(
                pandas_df.memory_usage(deep=True).sum()
            ),

            # clean_pandas()에서 만든 정제 결과를 추가합니다.
            **pandas_cleaning,
        },
        "polars": {
            # Polars의 행과 열 개수입니다.
            "rows": polars_df.height,
            "columns": polars_df.width,

            # 소수점 이하 6자리까지 로딩 시간을 저장합니다.
            "load_seconds": round(polars_seconds, 6),

            # Polars DataFrame이 사용하는 메모리 크기를 바이트 단위로 계산합니다.
            "memory_bytes": int(polars_df.estimated_size("b")),

            # clean_polars()에서 만든 정제 결과를 추가합니다.
            **polars_cleaning,
        },

        # 두 라이브러리의 최종 데이터 크기 일치 여부입니다.
        "same_cleaned_shape": same_shape,
    }
