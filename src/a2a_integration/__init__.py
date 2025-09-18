"""
A2A (Agent-to-Agent) Integration Module

LangGraph 에이전트와 A2A 프로토콜 간 통합을 단순화한 유틸리티/실행기/클라이언트
모음입니다. 최소한의 코드로 표준 A2A 서버를 띄우고, 클라이언트에서 안전하게
호출할 수 있도록 돕습니다.

Quick Start:
    # 서버 (LangGraph → A2A)
    from src.a2a_integration import (
        LangGraphAgentExecutor, create_agent_card, to_a2a_starlette_server
    )
    # graph = create_react_agent(...) or your compiled StateGraph
    app = to_a2a_starlette_server(
        graph=graph,
        agent_card=create_agent_card(
            name="MyAgent",
            description="Example agent",
            url="http://localhost:8000",
            skills=[],
        ),
    )

    # 클라이언트 (A2A → 에이전트)
    from src.a2a_integration import A2AClientManagerV2
    async with A2AClientManagerV2(base_url="http://localhost:8000") as client:
        text = await client.send_query("안녕하세요?")
        print(text)
"""

# Core executor
# Client utilities
from src.a2a_integration.a2a_lg_client_utils_v2 import (
    A2AClientManagerV2,
)

# Server utilities
from src.a2a_integration.a2a_lg_utils import (
    create_agent_card,
    to_a2a_run_uvicorn,
    to_a2a_starlette_server,
)
from src.a2a_integration.executor import LangGraphAgentExecutor

# Configuration
from src.a2a_integration.models import LangGraphExecutorConfig

__all__ = [
    # Core
    "LangGraphAgentExecutor",
    # Client utilities
    "A2AClientManagerV2",
    # Server utilities
    "create_agent_card",
    "to_a2a_run_uvicorn",
    "to_a2a_starlette_server",
    # Configuration
    "LangGraphExecutorConfig",
]
