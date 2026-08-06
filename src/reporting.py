# src/reporting.py
# 분석 결과를 Markdown 템플릿에 적용하여 최종 보고서를 생성하는 모듈입니다.
# Jinja2를 사용해 JSON 결과를 문서 형태로 렌더링하고 파일로 저장합니다.

# 파일과 폴더 경로를 운영체제에 맞게 처리합니다.
from pathlib import Path

# 분석 결과를 Markdown 템플릿에 적용합니다.
from jinja2 import Environment, FileSystemLoader, StrictUndefined


def generate_report(
    template_path: Path,
    output_path: Path,
    comparison: dict[str, object],
    eda: dict[str, object],
    statistics: dict[str, object],
    model_metrics: dict[str, object],
) -> None:
    """분석 결과를 Markdown 템플릿에 적용하여 보고서를 생성합니다."""

    # Jinja2 템플릿 환경을 설정합니다.
    environment = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Markdown 템플릿 파일을 불러옵니다.
    template = environment.get_template(template_path.name)

    # 템플릿 변수에 분석 결과를 전달합니다.
    rendered = template.render(
        comparison=comparison,
        eda=eda,
        statistics=statistics,
        model=model_metrics,
    )

    # 출력 폴더가 없으면 생성합니다.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 완성된 Markdown 보고서를 UTF-8 형식으로 저장합니다.
    output_path.write_text(
        rendered,
        encoding="utf-8",
    )
