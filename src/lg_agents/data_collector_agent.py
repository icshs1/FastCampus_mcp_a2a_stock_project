from typing import Any
from uuid import uuid4

import pytz
import structlog
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    convert_to_openai_messages,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from .util import load_env_file

logger = structlog.get_logger(__name__)

load_env_file()


async def create_data_collector_agent(
    model=None,
    is_debug: bool = False,
    checkpointer=None
):
    """
    create_react_agent를 통한 데이터 수집 에이전트

    MCP 도구를 로딩하고 프롬프트를 설정한 뒤, create_react_agent를 생성합니다.

    Args:
        model: 사용할 LLM 모델 (기본값: gpt-4.1-mini)
        is_debug: 디버그 모드 여부
        checkpointer: 체크포인터 (기본값: MemorySaver)

    Returns:
        create_react_agent로 생성된 LangGraph Agent

    사용 예:
        agent = await create_data_collector_agent()
        result = await agent.ainvoke({"messages": [...]})
    """
    try:
        # 1. MCP 도구 로딩
        from src.lg_agents.base.mcp_config import load_data_collector_tools

        from .prompts import get_prompt

        tools = await load_data_collector_tools()
        logger.info(f" Loaded {len(tools)} MCP tools for DataCollector")

        system_prompt = get_prompt("data_collector", "system", tool_count=len(tools))

        model = model or init_chat_model(
            model="gpt-4.1-mini",
            temperature=0, # NOTE: gpt-5 모델에서는 temperature 설정이 필요 없음 -> verbosity
            model_provider="openai",
        )

        checkpointer = checkpointer or MemorySaver()
        config = RunnableConfig(recursion_limit=10) # default: 25

        agent = create_react_agent(
            model=model,
            tools=tools,
            prompt=system_prompt,
            checkpointer=checkpointer,
            name="DataCollectorAgent",
            debug=is_debug,
            context_schema=config,
        )

        logger.info(" DataCollector Agent created successfully with create_react_agent")
        # NOTE: CompiledStateGraph 반환함으로써 바로 실행이 가능한 객체를 반환함
        return agent
    except Exception as e:
        logger.error(f"Failed to create DataCollector Agent: {e}")
        raise RuntimeError(f"DataCollector Agent creation failed: {e}") from e


async def collect_data(
    agent: CompiledStateGraph,
    symbols: list[str] = None,
    data_types: list[str] | None = None,
    user_question: str | None = None,
    context_id: str | None = None
) -> dict[str, Any]:
    """
    데이터 수집 실행 헬퍼 함수

    create_react_agent로 생성된 agent를 사용하여 데이터를 수집합니다.

    Args:
        agent: create_data_collector_agent()로 생성된 에이전트
        symbols: 수집할 종목 코드 리스트
        data_types: 수집할 데이터 타입 (선택적)
        user_question: 사용자 원본 질문 (선택적)
        context_id: 컨텍스트 ID (선택적)

    Returns:
        수집된 데이터 딕셔너리
    """
    try:
        data_types_str = ", ".join(data_types) if data_types else "모든 데이터"
        user_prompt = f"""종목 코드: {symbols or '없다면 질문을 통해서 기업명을 추출해서 찾아주세요.'}
        수집할 데이터: {data_types_str}
        질문: {user_question or '종합적인 데이터 수집'}

        위 종목들에 대한 데이터를 수집해주세요."""

        messages = [
            HumanMessage(content=user_prompt)
        ]

        result = await agent.ainvoke(
            {"messages": messages},
            config={
                "configurable": {
                    "thread_id": context_id or str(uuid4())
                }
            }
        )

        # Debug: print result structure
        logger.info(f"Debug - result type: {type(result)}")
        logger.info(f"Debug - result keys: {list(result.keys()) if hasattr(result, 'keys') else 'No keys'}")

        # create_react_agent 실행 결과에서 최종 AI 메시지 추출
        if "messages" not in result:
            logger.error(f" result에 'messages' 키가 없습니다. result: {result}")
            # Try to extract messages differently
            if hasattr(result, 'messages'):
                messages_list = result.messages
            else:
                messages_list = [result] if hasattr(result, 'content') else []
        else:
            messages_list = result["messages"]

        ai_messages = filter_messages(
            messages_list,
            include_types=[AIMessage],
        )

        if not ai_messages:
            logger.error("No AI messages found in the result")
            raise ValueError("No AI response generated")

        final_message: AIMessage = ai_messages[-1]

        try:
            from datetime import datetime

            # create_react_agent가 생성한 전체 메시지 히스토리 변환
            full_message_history = []
            msg_list: list[dict] = convert_to_openai_messages(messages_list)
            full_message_history.extend(msg_list)

            logger.info(f" create_react_agent 메시지 히스토리 구성 완료: {len(full_message_history)}개 메시지")
        except Exception as e:
            logger.error(f" create_react_agent 메시지 히스토리 구성 중 오류: {e}")
            full_message_history = []

        # create_react_agent가 수행한 도구 호출 횟수 계산
        tool_calls_made = sum(
            len(msg.tool_calls)
            for msg in filter_messages(messages_list, include_types=[AIMessage])
            if hasattr(msg, "tool_calls") and msg.tool_calls
        )

        logger.info(" create_react_agent 실행 완료 - 데이터 수집 요약:")
        logger.info(f"   → 총 도구 호출 횟수: {tool_calls_made}")
        logger.info(f"   → 총 메시지 수: {len(messages_list)}")
        logger.info(f"   → 수집된 종목: {symbols}")

        # 실행 결과 Dictionary 반환
        return {
            "success": True,
            "result": {
                "raw_response": final_message.content,
                "symbols_collected": symbols,
                "tool_calls_made": tool_calls_made,
                "total_messages_count": len(result["messages"]),
                "timestamp": datetime.now(tz=pytz.timezone("Asia/Seoul")).isoformat(),
            },
            "full_messages": full_message_history,
            "agent_type": "DataCollectorLangGraphAgent",
            "workflow_status": "completed",
            "error": None,
        }

    except Exception as e:
        logger.error(f" create_react_agent 기반 데이터 수집 실패: {e}")
        return {
            "success": False,
            "result": None,
            "error": str(e),
            "agent_type": "DataCollectorLangGraphAgent",
            "agent_implementation": "create_react_agent",
            "workflow_status": "failed",
        }
