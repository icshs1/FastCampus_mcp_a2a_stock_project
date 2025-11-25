"""
멀티 에이전트 Orchestrator SupervisorAgent

이 모듈의 SupervisorAgent 는 BaseGraphAgent 를 상속하며, 조건부 엣지를 사용해
하위 에이전트를 순차적으로 호출합니다. 또한 create_react_agent 의 Tool 기반
제약을 보완하고 비동기 A2A 호출을 지원합니다.

아키텍처:
    START → route → data_collector → analysis → trading → aggregate → END
    (각 단계는 워크플로우 패턴에 따라 조건부로 실행됩니다)
"""

from enum import Enum
from typing import Annotated, Any, ClassVar, Dict, List, Optional

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# 새로운 함수 기반 에이전트 import
from src.lg_agents.analysis_agent import analyze, create_analysis_agent  # noqa: E402
from src.lg_agents.base.base_graph_agent import BaseGraphAgent  # noqa: E402
from src.lg_agents.base.base_graph_state import BaseGraphState  # noqa: E402
from src.lg_agents.data_collector_agent import (  # noqa: E402
    collect_data,
    create_data_collector_agent,
)
from src.lg_agents.trading_agent import (  # noqa: E402
    create_trading_agent,
    execute_trading,
)

from .util import (  # noqa: E402
    extract_ai_messages_from_response,
    load_env_file,
)

load_env_file()
logger = structlog.get_logger(__name__)


class WorkflowPattern(str, Enum):
    """워크플로우 패턴 정의"""

    DATA_ONLY = "data_only"  # 데이터 수집만
    DATA_ANALYSIS = "data_analysis"  # 데이터 수집 + 분석
    FULL_WORKFLOW = "full_workflow"  # 데이터 수집 + 분석 + 거래


# ============================================
# State Definition
# ============================================


class SupervisorState(BaseGraphState):
    messages: Annotated[list[BaseMessage], add_messages]
    user_question: str = ""

    # Workflow Metadata
    workflow_pattern: Optional[WorkflowPattern] = None  # 워크플로우 패턴
    final_response: str = ""  # 최종 응답

    # 하위 에이전트 결과
    collected_data: Optional[Dict[str, Any]] = None  # DataCollectorAgent 결과
    analysis_result: Optional[Dict[str, Any]] = None  # AnalysisAgent 결과
    trading_result: Optional[Dict[str, Any]] = None  # TradingAgent 결과

    # Raw LLM decisions
    raw_routing_decision: Optional[str] = None  # LLM의 라우팅 판단
    success: bool = False  # 요청 성공 여부


# ============================================
# SupervisorAgent Implementation
# ============================================


class SupervisorAgent(BaseGraphAgent):
    """Sequential Multi-Node SupervisorAgent - 하위 에이전트 오케스트레이션, LangGraph 통합 인터페이스 제공"""

    NODE_NAMES: ClassVar[Dict[str, str]] = {
        "ROUTING": "route",
        "DATA_COLLECTOR": "data_collector",
        "ANALYSIS": "analysis",
        "TRADING": "trading",
        "AGGREGATE": "aggregate",
    }

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        is_debug: bool = True,
    ):
        """SupervisorAgent"""
        if model is None:
            self.model = ChatOpenAI(
                model="gpt-4.1-mini",
                temperature=0,
            )

        if checkpointer is None:
            self.checkpointer = MemorySaver()

        self.is_debug = is_debug

        super().__init__(
            model=self.model,
            checkpointer=self.checkpointer,
            state_schema=SupervisorState,
            is_debug=self.is_debug,
            agent_name="LangGraphSupervisorAgent",
        )

    def init_nodes(self, graph: StateGraph):
        graph.add_node(self.get_node_name("ROUTING"), self._route_request)
        graph.add_node(self.get_node_name("DATA_COLLECTOR"), self._call_data_collector)
        graph.add_node(self.get_node_name("ANALYSIS"), self._call_analysis)
        graph.add_node(self.get_node_name("TRADING"), self._call_trading)

    def init_edges(self, graph: StateGraph):
        graph.add_edge(START, self.get_node_name("ROUTING"))
        # 라우팅 후 조건부 엣지
        graph.add_conditional_edges(self.get_node_name("ROUTING"), self._get_next_step)

        # DataCollector 후 조건부 엣지
        graph.add_conditional_edges(
            self.get_node_name("DATA_COLLECTOR"),
            self._get_next_after_data,
        )

        # Analysis 후 조건부 엣지
        graph.add_conditional_edges(
            self.get_node_name("ANALYSIS"),
            self._get_next_after_analysis,
        )
        graph.add_edge(self.get_node_name("TRADING"), END)

    # ============================================
    # Node Functions
    # ============================================

    async def _route_request(
        self,
        state: SupervisorState,
        config: RunnableConfig,
    ) -> SupervisorState:
        """사용자 요청을 분석하여 워크플로우 패턴 결정"""
        logger.info(f"[SupervisorAgent] Routing request: {state['messages']}")
        try:
            # 마지막 메시지 추출
            filtered_messages = filter_messages(
                state["messages"], include_types=[HumanMessage]
            )
            last_message = filtered_messages[-1] if filtered_messages else None
            if not last_message:
                logger.error("No user message found")
                return state

            state["user_question"] = last_message.content

            # LLM을 사용한 워크플로우 결정
            routing_prompt = f"""사용자 요청을 분석하여 적절한 워크플로우를 결정하세요.

사용자 요청: {state["user_question"]}

[워크플로우 패턴]
1. DATA_ONLY: 단순 데이터 조회 (주가, 뉴스, 정보 조회)
2. DATA_ANALYSIS: 데이터 수집 + 분석 (투자 분석, 평가)
3. FULL_WORKFLOW: 데이터 수집 + 분석 + 거래 (매수/매도 실행)

패턴명만 영어 대문자로 응답하세요."""

            response = await self.model.ainvoke([HumanMessage(content=routing_prompt)])
            # LangChain AIMessage.content는 str | list 타입이지만, ChatOpenAI는 항상 str 반환
            content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            decision = content.strip().upper()  # 대문자로 변환

            # 워크플로우 패턴 매핑
            if decision == "DATA_ONLY":
                state["workflow_pattern"] = WorkflowPattern.DATA_ONLY
            elif decision == "DATA_ANALYSIS":
                state["workflow_pattern"] = WorkflowPattern.DATA_ANALYSIS
            elif decision == "FULL_WORKFLOW":
                state["workflow_pattern"] = WorkflowPattern.FULL_WORKFLOW
            else:
                if "주가" in state["user_question"] or "가격" in state["user_question"]:
                    state["workflow_pattern"] = WorkflowPattern.DATA_ONLY
                else:
                    state["workflow_pattern"] = WorkflowPattern.DATA_ANALYSIS

            state["raw_routing_decision"] = decision
            state["messages"] = [
                AIMessage(content=f"워크플로우 패턴 결정: {state['workflow_pattern']}")
            ]
            state["success"] = True  # 요청 성공

            logger.info(f" Workflow pattern: {state['workflow_pattern']}")
            return state

        except Exception as e:
            logger.error(f"Routing error: {e}")
            state["workflow_pattern"] = None  # 라우팅 실패 시 None으로 설정
            state["success"] = False  # 요청 실패
            return state

    async def _call_data_collector(
        self,
        state: SupervisorState,
        config: RunnableConfig,
    ) -> SupervisorState:
        """DataCollectorAgent 호출"""
        try:
            input_data = {
                "symbols": [],
                "data_types": ["news", "price", "info"],
                "user_question": state["user_question"],
            }

            langgraph_agent = await create_data_collector_agent(is_debug=False)
            response = await collect_data(
                agent=langgraph_agent,
                symbols=input_data["symbols"],
                data_types=input_data["data_types"],
                user_question=input_data["user_question"],
                context_id=config.get("configurable", {}).get("thread_id", None),
            )
            logger.info(f"[SupervisorAgent] DataCollectorAgent response: {response}")
            ai_messages = extract_ai_messages_from_response(response)

            state["collected_data"] = response
            # ai_messages가 비어있지 않은 경우에만 메시지 추가
            if ai_messages:
                state["messages"] = [ai_messages[-1]]
                state["success"] = True  # 요청 성공
            else:
                # 오류가 발생한 경우 오류 메시지를 AIMessage로 생성
                error_msg = response.get("error", "데이터 수집 중 오류가 발생했습니다.")
                state["messages"] = [
                    AIMessage(content=f" DataCollectorAgent 오류: {error_msg}")
                ]
                state["success"] = False  # 요청 실패

            logger.info(" DataCollectorAgent completed")
            return state

        except Exception as e:
            logger.error(f"DataCollectorAgent error: {e}")
            state["collected_data"] = {"error": str(e)}
            state["success"] = False  # 요청 실패
            return state

    async def _call_analysis(
        self,
        state: SupervisorState,
        config: RunnableConfig,
    ) -> SupervisorState:
        """AnalysisAgent 호출"""
        try:
            # 데이터 준비
            input_data = {
                "symbols": self._extract_symbols(state),
                "collected_data": state["collected_data"] or {},
                "user_question": state["user_question"],
            }

            langgraph_agent = await create_analysis_agent(is_debug=False)
            response = await analyze(
                agent=langgraph_agent,
                symbols=input_data["symbols"],
                collected_data=input_data["collected_data"],
                user_question=input_data["user_question"],
                context_id=config.get("configurable", {}).get("thread_id", None),
            )
            logger.info(f"[SupervisorAgent] AnalysisAgent response: {response}")
            ai_messages = extract_ai_messages_from_response(response)

            state["analysis_result"] = response
            # ai_messages가 비어있지 않은 경우에만 메시지 추가
            if ai_messages:
                state["messages"] = [ai_messages[-1]]
                state["success"] = True  # 요청 성공
            else:
                # 오류가 발생한 경우 오류 메시지를 AIMessage로 생성
                error_msg = response.get("error", "분석 중 오류가 발생했습니다.")
                state["messages"] = [
                    AIMessage(content=f" AnalysisAgent 오류: {error_msg}")
                ]
                state["success"] = False  # 요청 실패

            logger.info(" AnalysisAgent completed")
            return state

        except Exception as e:
            logger.error(f"AnalysisAgent error: {e}")
            state["analysis_result"] = {"error": str(e)}
            state["success"] = False  # 요청 실패
            return state

    async def _call_trading(
        self,
        state: SupervisorState,
        config: RunnableConfig,
    ) -> SupervisorState:
        """TradingAgent 호출"""
        try:
            # 분석 결과에서 거래 신호 추출
            trading_signal = self._extract_trading_signal(state)

            # 데이터 준비
            input_data = {
                "symbols": self._extract_symbols(state),
                "trading_signal": trading_signal,
                "analysis_result": state["analysis_result"] or {},
                "user_question": state["user_question"],
            }

            langgraph_agent = await create_trading_agent(is_debug=False)
            response = await execute_trading(
                agent=langgraph_agent,
                analysis_result=input_data["analysis_result"],
                user_question=input_data["user_question"],
                context_id=config.get("configurable", {}).get("thread_id", None),
            )
            logger.info(f"[SupervisorAgent] TradingAgent response: {response}")
            ai_messages = extract_ai_messages_from_response(response)

            state["trading_result"] = response
            # ai_messages가 비어있지 않은 경우에만 메시지 추가
            if ai_messages:
                state["messages"] = [ai_messages[-1]]
                state["success"] = True  # 요청 성공
            else:
                # 오류가 발생한 경우 오류 메시지를 AIMessage로 생성
                error_msg = response.get("error", "거래 실행 중 오류가 발생했습니다.")
                state["messages"] = [
                    AIMessage(content=f" TradingAgent 오류: {error_msg}")
                ]
                state["success"] = False  # 요청 실패
            logger.info(" TradingAgent completed")
            return state

        except Exception as e:
            logger.error(f"TradingAgent error: {e}")
            state["trading_result"] = {"error": str(e)}
            state["success"] = False  # 요청 실패
            return state

    # ============================================
    # Conditional Edge Functions
    # ============================================

    def _get_next_step(self, state: SupervisorState):
        """라우팅 후 다음 단계 결정"""
        logger.info(f"[SupervisorAgent] Routing request: {state}")
        if not state.get("workflow_pattern"):
            return END
        return self.get_node_name("DATA_COLLECTOR")

    def _get_next_after_data(self, state: SupervisorState):
        """DataCollector 후 다음 단계 결정"""
        if state.get("workflow_pattern") is None:
            return END
        return self.get_node_name("ANALYSIS")

    def _get_next_after_analysis(self, state: SupervisorState):
        """Analysis 후 다음 단계 결정"""
        if state.get("workflow_pattern") is None:
            return END
        return self.get_node_name("TRADING")

    # ============================================
    # Helper Functions
    # ============================================

    def _extract_symbols(self, state: SupervisorState) -> List[str]:
        """수집된 데이터에서 종목 코드 추출"""
        symbols = []
        # collected_data에서 추출
        if state["collected_data"] and isinstance(state["collected_data"], dict):
            data = state["collected_data"].get("collected_data", {})
            if "symbols_processed" in data:
                symbols = data["symbols_processed"]
        return symbols if symbols else ["005930"]  # 기본값: 삼성전자

    def _extract_trading_signal(self, state: SupervisorState) -> str:
        """분석 결과에서 거래 신호 추출"""
        if state["analysis_result"] and isinstance(state["analysis_result"], dict):
            analysis = state["analysis_result"].get("analysis_result", {})
            if "investment_signal" in analysis:
                return analysis["investment_signal"]
        return "HOLD"  # 기본값
