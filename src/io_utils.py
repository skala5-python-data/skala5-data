# src/io_utils.py

# Python 객체를 JSON 형식으로 저장하기 위해 사용하는 표준 라이브러리입니다.
import json

# 운영체제와 관계없이 파일 및 폴더 경로를 안전하게 다루기 위해 사용합니다.
from pathlib import Path

# 여러 자료형을 함수의 매개변수로 받을 수 있도록 Any 타입을 사용합니다.
from typing import Any


def ensure_parent(path: Path) -> None:
    """파일을 저장하기 전에 상위 폴더가 존재하도록 생성합니다."""

    # path.parent는 파일이 저장될 상위 폴더를 의미합니다.
    # parents=True는 중간 폴더가 없어도 함께 생성합니다.
    # exist_ok=True는 폴더가 이미 존재해도 오류를 발생시키지 않습니다.
    path.parent.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    """기본 JSON 변환이 불가능한 값을 변환 가능한 형식으로 바꿉니다."""

    # NumPy의 int64, float64 같은 자료형은 json.dump()가 바로 처리하지 못합니다.
    # item() 메서드가 있으면 Python의 기본 int 또는 float 자료형으로 변환합니다.
    if hasattr(value, "item"):
        return value.item()

    # Path 객체도 JSON에 직접 저장할 수 없으므로 문자열 경로로 변환합니다.
    if isinstance(value, Path):
        return str(value)

    # 위 조건으로 처리할 수 없는 자료형이면 명확한 오류를 발생시킵니다.
    raise TypeError(
        f"JSON으로 변환할 수 없는 형식입니다: {type(value)!r}"
    )


def save_json(data: Any, path: Path) -> None:
    """Python 데이터를 지정한 경로에 JSON 파일로 저장합니다."""

    # JSON 파일을 저장할 상위 폴더가 없으면 먼저 생성합니다.
    ensure_parent(path)

    # UTF-8 인코딩으로 파일을 쓰기 모드로 엽니다.
    # with 문을 사용하면 작업이 끝난 뒤 파일이 자동으로 닫힙니다.
    with path.open("w", encoding="utf-8") as file:

        # Python 객체를 JSON 형식으로 변환하여 파일에 저장합니다.
        json.dump(
            data,
            file,

            # 한글을 \uXXXX 형식으로 바꾸지 않고 그대로 저장합니다.
            ensure_ascii=False,

            # 들여쓰기 2칸을 적용하여 사람이 읽기 쉬운 형태로 저장합니다.
            indent=2,

            # JSON이 처리하지 못하는 자료형은 json_default() 함수로 변환합니다.
            default=json_default,
        )