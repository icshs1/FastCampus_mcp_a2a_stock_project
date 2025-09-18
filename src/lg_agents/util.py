import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List

from langchain_core.messages import AIMessage, convert_to_messages, filter_messages


@lru_cache(1)
def load_env_file():
    """프로젝트 루트의 .env 파일을 로드.

    초보자 메모:
        - `.env` 파일은 개발 환경 변수들을 저장합니다.
        - 운영 환경에서는 안전한 비밀 관리(예: 환경변수/비밀관리)를 권장합니다.
    """
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip("'\"")
                    os.environ[key] = value


def extract_ai_messages_from_response(react_agent_response: dict[str, Any]) -> List[AIMessage]:
    """응답에서 AI 메시지를 추출하는 공통 함수.

    LangGraph/React 에이전트의 결과는 다양한 포맷으로 반환될 수 있으므로,
    이 함수는 ``full_messages``를 OpenAI 호환 메시지로 변환한 뒤 AI 메시지만
    필터링합니다.

    Args:
        react_agent_response: 에이전트 응답 딕셔너리 (``full_messages`` 키 포함 예상)

    Returns:
        list[AIMessage]: 추출된 AI 메시지 리스트 (없으면 빈 리스트)
    """
    original_messages = react_agent_response.get("full_messages", [])
    converted_messages = convert_to_messages(original_messages)
    ai_messages = filter_messages(converted_messages, include_types=[AIMessage])
    return ai_messages
