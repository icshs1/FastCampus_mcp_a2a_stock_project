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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from .util import load_env_file

load_env_file()

logger = structlog.get_logger(__name__)

async def create_trading_agent(model=None, is_debug: bool = False):
    """Trading Agent 생성

    Args:
        model: LLM 모델 (기본값: gpt-5-mini)
        is_debug: 디버그 모드 여부

    Returns:
        create_react_agent: 바로 사용 가능한 react agent
    """
    try:
        # LLM 모델 초기화
        llm_model = model or init_chat_model(
            model="gpt-4.1-mini",
            model_provider="openai",
        )

        # MCP 도구 로딩
        from src.lg_agents.base.mcp_config import load_trading_tools

        from .prompts import get_prompt

        tools = await load_trading_tools()
        logger.info(f"Loaded {len(tools)} MCP tools for React TradingAgent")

        system_prompt = get_prompt("trading", "system", tool_count=len(tools))

        check_pointer = MemorySaver()

        # TODO: _human_in_the_loop 함수 적용을 어디다가 해야할까?
        # interrupt 를 어느 노드(또는 위치)에 할지 고민
        agent = create_react_agent(
            model=llm_model,
            tools=tools,
            prompt=system_prompt,
            checkpointer=check_pointer,
            name="LangGraphTradingAgent",
            debug=is_debug,
        )

        logger.info(" create_react_agent 기반 TradingAgent 생성 완료")
        return agent

    except Exception as e:
        logger.error(f"Failed to create React TradingAgent: {e}")
        raise

async def execute_trading(
    agent: CompiledStateGraph,
    analysis_result: dict[str, Any],
    user_question: str | None = None,
    context_id: str | None = None
) -> dict[str, Any]:
    """
    TradingAgent 을 통한 거래 실행 함수

    Args:
        agent: create_react_agent로 생성된 agent
        analysis_result: AnalysisAgent 분석 결과
        user_question: 사용자 원본 질문
        context_id: 컨텍스트 ID

    Returns:
        거래 실행 결과 딕셔너리
    """
    try:
        symbols = analysis_result.get("symbols", [])
        trading_signal = analysis_result.get("trading_signal", "HOLD")
        user_question = user_question or "거래 실행"

        trading_prompt = f"""
[거래 요청]
- 거래 대상 종목: {symbols}
- 거래 신호: {trading_signal}
- 사용자 요청: {user_question}

[분석 결과 정보]
{analysis_result}

[거래 실행 단계]

1. 컨텍스트 분석:
   - 현재 시장 상황 및 투자 환경 파악
   - 사용자 투자 목적 및 리스크 성향 분석
   - 거래 신호의 신뢰도 및 타이밍 검증

2. 전략 수립:
   - 분석 결과를 바탕으로 최적 투자 전략 선택
   - MOMENTUM/VALUE/BALANCED 중 적합한 전략 결정
   - 투자 기간 및 목표 수익률 설정

3. 포트폴리오 최적화:
   - 포지션 크기 및 배분 최적화
   - 단일 종목 20% 한도 준수 확인

4. 리스크 평가:
   - VaR 95% 신뢰수준 계산 (도구 사용)
   - 리스크 점수 산출 (0-1 스케일)
   - 스톱로스/테이크프로핏 레벨 설정

5. 승인 처리:
   - 리스크 점수 기반 승인 필요성 판단
   - 고위험(>0.7) 거래시 Human 승인 대기
   - 자동 실행 조건 확인

6. 주문 실행:
    도구 리스트: place_buy_order, place_sell_order, modify_order
   - 주문 타입 (시장가/지정가) 결정
   - 체결 확인 및 결과 기록

7. 모니터링:
   - 주문 상태 실시간 추적
   - 포트폴리오 성과 업데이트
   - 리스크 메트릭 재계산

[실행 방식]
- 반드시 사용 가능한 도구들을 활용하여 실제 데이터 기반 의사결정
- 추측이나 가정이 아닌 계산된 리스크 메트릭 사용
- 모든 거래 결정과 근거를 상세히 기록
"""

        messages = [HumanMessage(content=trading_prompt)]

        result = await agent.ainvoke(
            {"messages": messages},
            config={
                "configurable": {
                    "thread_id": context_id or str(uuid4())
                }
            }
        )

        # create_react_agent 실행 결과에서 최종 AI 메시지 추출
        ai_messages = filter_messages(
            result["messages"],
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
            msg_list: list[dict] = convert_to_openai_messages(result["messages"])
            full_message_history.extend(msg_list)

            logger.info(f" create_react_agent 메시지 히스토리 구성 완료: {len(full_message_history)}개 메시지")
        except Exception as e:
            logger.error(f" create_react_agent 메시지 히스토리 구성 중 오류: {e}")
            full_message_history = []

        tool_calls_made = sum(
            len(msg.tool_calls)
            for msg in filter_messages(result["messages"], include_types=[AIMessage])
            if hasattr(msg, "tool_calls") and msg.tool_calls
        )

        return {
            "success": True,
            "result": {
                "raw_trading": final_message.content,
                "symbols_traded": symbols,
                "trading_signal": trading_signal,
                "tool_calls_made": tool_calls_made,
                "total_messages_count": len(result["messages"]),
                "timestamp": datetime.now(tz=pytz.timezone("Asia/Seoul")).isoformat(),
            },
            "full_messages": full_message_history,
            "agent_type": "TradingLangGraphAgent",
            "workflow_status": "completed",
            "error": None,
        }

    except Exception as e:
        logger.error(f"Trading execution failed: {e}")
        return {
            "success": False,
            "result": None,
            "error": str(e),
            "agent_type": "TradingLangGraphAgent",
            "workflow_status": "failed",
        }
