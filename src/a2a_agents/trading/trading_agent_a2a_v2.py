"""
A2A 통합을 갖춘 트레이딩 에이전트.

이 모듈은 위험 관리에 대한 Human-in-the-Loop 승인을 통해 표준화된
A2A 인터페이스를 구현하는 트레이딩 에이전트를 제공합니다.

참고사항:
    - 승인 절차: 고위험 작업의 경우 ``status='input_required'``와 함께
      승인 요청 페이로드를 반환합니다. 클라이언트는 후속 메시지로
      "approve/승인" 또는 "reject/거부"로 응답해야 합니다.
    - 위험 임계값: ``risk_threshold``는 승인이 필요한 시점을 결정합니다.
      추가 안전장치로는 대량 주문 감지와 동시 보류 주문 수가 있습니다.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

import pytz
import structlog
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.lg_agents.base.a2a_interface import A2AOutput, A2AStreamBuffer, BaseA2AAgent
from src.lg_agents.base.base_graph_agent import BaseGraphAgent
from src.lg_agents.base.mcp_config import load_trading_tools
from src.lg_agents.prompts import get_prompt
from src.lg_agents.util import load_env_file

logger = structlog.get_logger(__name__)

load_env_file()


class TradingA2AAgent(BaseA2AAgent, BaseGraphAgent):
    """
    A2A 통합 및 Human-in-the-Loop 지원을 갖춘 트레이딩 에이전트.

    이 에이전트는 다음을 처리합니다:
    - 주문 실행 및 관리
    - 위험 평가 (VaR, 샤프 비율)
    - 포트폴리오 최적화
    - 고위험 거래에 대한 인간 승인 워크플로
    """

    def __init__(
        self,
        model=None,
        is_debug: bool = False,
        checkpointer=None,
        risk_threshold: float = 0.15  # 15% VaR threshold
    ):
        """
        트레이딩 A2A 에이전트를 초기화합니다.

        Args:
            model: LLM 모델 (기본값: gpt-4.1-mini)
            is_debug: 디버그 모드 플래그
            checkpointer: 체크포인트 관리자 (기본값: MemorySaver)
            risk_threshold: 승인이 필요한 VaR 임계값
        """
        BaseA2AAgent.__init__(self)

        # 모델 초기화
        self.model = model or init_chat_model(
            model="gpt-4.1-mini",
            temperature=0,
            model_provider="openai"
        )
        self.checkpointer = checkpointer or MemorySaver()
        self.risk_threshold = risk_threshold

        # 필요한 매개변수로 BaseGraphAgent 초기화
        BaseGraphAgent.__init__(
            self,
            model=self.model,
            checkpointer=self.checkpointer,
            is_debug=is_debug,
            lazy_init=True,  # Use lazy initialization for A2A agents
            agent_name="TradingA2AAgent"
        )

        self.tools = []

        # LLM 출력 관리를 위한 스트림 버퍼
        self.stream_buffer = A2AStreamBuffer(max_size=100)

        # 트레이딩 상태 추적
        self.pending_orders = []
        self.executed_trades = []
        self.portfolio_state = {}
        self.risk_metrics = {
            "var_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0
        }
        self.requires_approval = False
        self.approval_status = "PENDING"

    async def initialize(self):
        """MCP 도구로 에이전트를 초기화하고 그래프를 생성합니다.

        Raises:
            RuntimeError: 도구를 로드할 수 없거나 그래프 생성에 실패한 경우
        """
        try:
            # MCP 도구 로드
            self.tools = await load_trading_tools()
            logger.info(f" Loaded {len(self.tools)} MCP tools for Trading")

            # 시스템 프롬프트 가져오기
            system_prompt = get_prompt("trading", "system", tool_count=len(self.tools))

            # 반응형 에이전트 그래프 생성
            config = RunnableConfig(recursion_limit=10)
            self.graph = create_react_agent(
                model=self.model,
                tools=self.tools,
                prompt=system_prompt,
                checkpointer=self.checkpointer,
                name="TradingAgent",
                debug=self.is_debug,
                context_schema=config
            )

            logger.info(" Trading A2A Agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Trading Agent: {e}")
            raise RuntimeError(f"Trading Agent initialization failed: {e}") from e

    async def execute_for_a2a(
        self,
        input_dict: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> A2AOutput:
        """
        A2A 호환 입력 및 출력으로 에이전트를 실행합니다.

        고위험 거래에 대한 Human-in-the-Loop 승인을 포함합니다.

        Args:
            input_dict: 트레이딩 요청 (보통 ``{"messages": [...]}`` 형태)
            config: 선택적 실행 설정. 생략 시 기본 ``thread_id`` 사용

        Returns:
            A2AOutput: A2A 처리를 위한 표준화된 출력
        """
        if not self.graph:
            await self.initialize()

        try:
            # 추적 변수 초기화
            self.pending_orders.clear()
            self.executed_trades.clear()
            self.requires_approval = False
            self.approval_status = "PENDING"

            # 재개 작업인지 확인 (인간 응답)
            if self._is_approval_response(input_dict):
                return await self._handle_approval_response(input_dict)

            # 그래프 실행
            result = await self.graph.ainvoke(
                input_dict,
                config=config or {"configurable": {"thread_id": str(uuid4())}}
            )

            # 승인 필요 여부 확인
            if self._requires_human_approval():
                return self._create_approval_request()

            # 최종 출력 추출
            return self.extract_final_output(result)

        except Exception as e:
            return self.format_error(e, context="execute_for_a2a")

    def format_stream_event(
        self,
        event: Dict[str, Any]
    ) -> Optional[A2AOutput]:
        """LangGraph 이벤트를 A2A 스트리밍 업데이트로 변환합니다.

        주문/위험/포트폴리오 단계에 대해 사용자 친화적인 텍스트를 출력합니다.
        """
        event_type = event.get("event", "")

        # LLM 스트리밍 처리
        if event_type == "on_llm_stream":
            content = self.extract_llm_content(event)
            if content and self.stream_buffer.add(content):
                # 버퍼가 가득 찼으므로 비우기
                return self.create_a2a_output(
                    status="working",
                    text_content=self.stream_buffer.flush(),
                    stream_event=True,
                    metadata={"event_type": "llm_stream"}
                )

        # 도구 실행 이벤트 처리
        elif event_type == "on_tool_start":
            tool_name = event.get("name", "unknown")

            # 다양한 도구 유형 추적
            if "order" in tool_name.lower():
                return self.create_a2a_output(
                    status="working",
                    text_content=f" 주문 준비 중: {tool_name}",
                    stream_event=True,
                    metadata={
                        "event_type": "tool_start",
                        "tool_category": "order",
                        "tool_name": tool_name
                    }
                )
            elif "risk" in tool_name.lower() or "var" in tool_name.lower():
                return self.create_a2a_output(
                    status="working",
                    text_content=f"️ 리스크 평가 중: {tool_name}",
                    stream_event=True,
                    metadata={
                        "event_type": "tool_start",
                        "tool_category": "risk",
                        "tool_name": tool_name
                    }
                )
            elif "portfolio" in tool_name.lower():
                return self.create_a2a_output(
                    status="working",
                    text_content=f" 포트폴리오 분석 중: {tool_name}",
                    stream_event=True,
                    metadata={
                        "event_type": "tool_start",
                        "tool_category": "portfolio",
                        "tool_name": tool_name
                    }
                )

        # 결과와 함께 도구 완료 처리
        elif event_type == "on_tool_end":
            tool_output = event.get("data", {}).get("output", {})
            tool_name = event.get("name", "unknown")

            # 위험 평가 결과 처리
            if "risk" in tool_name.lower() or "var" in tool_name.lower():
                if isinstance(tool_output, dict):
                    var_ratio = tool_output.get("var_ratio", 0)
                    self.risk_metrics["var_ratio"] = var_ratio

                    # 승인 필요 여부 확인
                    if var_ratio > self.risk_threshold:
                        self.requires_approval = True
                        return self.create_a2a_output(
                            status="working",
                            text_content=f"️ 높은 리스크 감지 (VaR: {var_ratio:.2%}). 승인이 필요합니다.",
                            data_content={
                                "risk_assessment": tool_output,
                                "requires_approval": True
                            },
                            stream_event=True,
                            metadata={"event_type": "risk_warning"}
                        )

            # 주문 결과 처리
            elif "order" in tool_name.lower():
                if isinstance(tool_output, dict):
                    order_id = tool_output.get("order_id")
                    if order_id:
                        self.pending_orders.append(tool_output)
                        return self.create_a2a_output(
                            status="working",
                            data_content={
                                "order_created": tool_output,
                                "pending_orders_count": len(self.pending_orders)
                            },
                            stream_event=True,
                            metadata={"event_type": "order_created"}
                        )

        # 완료 이벤트 처리
        elif self.is_completion_event(event):
            # 남은 버퍼 내용 비우기
            if self.stream_buffer.has_content():
                return self.create_a2a_output(
                    status="working",
                    text_content=self.stream_buffer.flush(),
                    stream_event=True,
                    metadata={"event_type": "buffer_flush"}
                )

        return None

    def extract_final_output(
        self,
        state: Dict[str, Any]
    ) -> A2AOutput:
        """위험 및 승인 필드를 포함한 최종 ``A2AOutput``을 생성합니다."""
        try:
            # 상태에서 메시지 추출
            messages = state.get("messages", [])

            # 마지막 AI 메시지를 트레이딩 요약으로 가져오기
            trading_summary = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
                    trading_summary = msg.content
                    break

            # 트레이딩 신호 결정
            trading_signal = self._extract_trading_action(trading_summary)

            # 총 주문 금액 계산
            total_amount = sum(
                order.get("amount", 0) for order in self.pending_orders
            )

            # 구조화된 데이터 준비
            data_content = {
                "success": True,
                "result": {
                    "raw_trading": trading_summary,
                    "symbols_traded": self._extract_symbols_from_orders(),
                    "trading_signal": trading_signal,
                    "order_amount": total_amount,
                    "orders_pending": len(self.pending_orders),
                    "orders_executed": len(self.executed_trades),
                    "risk_assessment": {
                        "var_ratio": self.risk_metrics["var_ratio"],
                        "sharpe_ratio": self.risk_metrics["sharpe_ratio"],
                        "max_drawdown": self.risk_metrics["max_drawdown"],
                        "requires_approval": self.requires_approval
                    },
                    "approval_status": self.approval_status,
                    "timestamp": datetime.now(pytz.UTC).isoformat()
                },
                "agent_type": "TradingA2AAgent",
                "workflow_status": "completed" if not self.requires_approval else "input_required"
            }

            # 최종 상태 결정
            status = "input_required" if self.requires_approval else "completed"

            # 최종 출력 생성
            return self.create_a2a_output(
                status=status,
                text_content=trading_summary or "거래 준비가 완료되었습니다.",
                data_content=data_content,
                final=not self.requires_approval,
                requires_approval=self.requires_approval,
                metadata={
                    "execution_complete": not self.requires_approval,
                    "pending_orders": len(self.pending_orders),
                    "risk_level": self._determine_risk_level()
                }
            )

        except Exception as e:
            logger.error(f"Error extracting final output: {e}")
            return self.format_error(e, context="extract_final_output")

    # Human-in-the-Loop methods

    def _requires_human_approval(self) -> bool:
        """진행하기 전에 인간의 승인이 필요한 경우 True를 반환합니다."""
        # VaR 임계값 확인
        if self.risk_metrics["var_ratio"] > self.risk_threshold:
            return True

        # 대량 주문 확인
        total_amount = sum(order.get("amount", 0) for order in self.pending_orders)
        if total_amount > 10000000:  # 10M KRW threshold
            return True

        # 다중 주문 확인
        if len(self.pending_orders) > 5:
            return True

        return False

    def _create_approval_request(self) -> A2AOutput:
        """인간 검토를 위한 승인 요청 출력을 생성합니다."""
        approval_message = self._build_approval_message()

        return self.create_a2a_output(
            status="input_required",
            text_content=approval_message,
            data_content={
                "approval_request": {
                    "pending_orders": self.pending_orders,
                    "risk_metrics": self.risk_metrics,
                    "total_amount": sum(order.get("amount", 0) for order in self.pending_orders),
                    "reason": self._get_approval_reason()
                }
            },
            requires_approval=True,
            metadata={
                "approval_type": "trading",
                "risk_level": self._determine_risk_level()
            }
        )

    def _build_approval_message(self) -> str:
        """승인 UI를 위한 사용자 대상 메시지 내용을 생성합니다."""
        total_amount = sum(order.get("amount", 0) for order in self.pending_orders)
        var_ratio = self.risk_metrics["var_ratio"]

        message = " **거래 승인 요청**\n\n"
        message += f" 주문 개수: {len(self.pending_orders)}개\n"
        message += f" 총 거래 금액: {total_amount:,.0f}원\n"
        message += f"️ VaR 비율: {var_ratio:.2%}\n"
        message += f" 리스크 수준: {self._determine_risk_level()}\n\n"

        message += "**주문 내역:**\n"
        for i, order in enumerate(self.pending_orders, 1):
            symbol = order.get("symbol", "N/A")
            action = order.get("action", "N/A")
            amount = order.get("amount", 0)
            message += f"{i}. {symbol} - {action} - {amount:,.0f}원\n"

        message += "\n승인하시려면 'approve' 또는 '승인'을,\n"
        message += "거부하시려면 'reject' 또는 '거부'를 입력해주세요."

        return message

    def _is_approval_response(self, input_dict: Dict[str, Any]) -> bool:
        """최신 메시지에 승인 결정이 포함되어 있는지 감지합니다."""
        messages = input_dict.get("messages", [])
        if not messages:
            return False

        last_message = messages[-1]
        if hasattr(last_message, "content"):
            content = last_message.content.lower()
            return any(word in content for word in ["approve", "승인", "reject", "거부", "cancel", "취소"])

        return False

    async def _handle_approval_response(self, input_dict: Dict[str, Any]) -> A2AOutput:
        """사용자의 승인/거부 결정을 적용하고 출력을 마무리합니다."""
        messages = input_dict.get("messages", [])
        response = messages[-1].content.lower() if messages else ""

        if "approve" in response or "승인" in response:
            self.approval_status = "APPROVED"
            # 보류 중인 주문 실행
            for order in self.pending_orders:
                self.executed_trades.append(order)

            return self.create_a2a_output(
                status="completed",
                text_content=" 거래가 승인되어 실행되었습니다.",
                data_content={
                    "approval_status": "APPROVED",
                    "executed_trades": self.executed_trades
                },
                final=True,
                metadata={"approval_processed": True}
            )
        else:
            self.approval_status = "REJECTED"
            return self.create_a2a_output(
                status="completed",
                text_content=" 거래가 거부되었습니다.",
                data_content={
                    "approval_status": "REJECTED",
                    "cancelled_orders": self.pending_orders
                },
                final=True,
                metadata={"approval_processed": True}
            )

    # 도우미 메서드

    def _extract_symbols_from_orders(self) -> list[str]:
        """보류 중인/실행된 주문에서 참조되는 고유한 심볼 집합을 반환합니다."""
        symbols = set()
        for order in self.pending_orders + self.executed_trades:
            if "symbol" in order:
                symbols.add(order["symbol"])
        return list(symbols)

    def _extract_trading_action(self, text: str) -> str:
        """비구조화된 텍스트에서 매수/매도/보유 의도를 파싱합니다."""
        text_lower = text.lower()

        if "매수" in text or "buy" in text_lower:
            return "BUY"
        elif "매도" in text or "sell" in text_lower:
            return "SELL"
        elif "보유" in text or "hold" in text_lower:
            return "HOLD"

        return "PENDING"

    def _determine_risk_level(self) -> str:
        """VaR 비율을 질적 위험 등급으로 매핑합니다."""
        var_ratio = self.risk_metrics["var_ratio"]

        if var_ratio > 0.20:
            return "HIGH"
        elif var_ratio > 0.10:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_approval_reason(self) -> str:
        """현재 상태에서 승인이 필요한 이유를 설명합니다."""
        reasons = []

        if self.risk_metrics["var_ratio"] > self.risk_threshold:
            reasons.append(f"High VaR ratio: {self.risk_metrics['var_ratio']:.2%}")

        total_amount = sum(order.get("amount", 0) for order in self.pending_orders)
        if total_amount > 10000000:
            reasons.append(f"Large order amount: {total_amount:,.0f} KRW")

        if len(self.pending_orders) > 5:
            reasons.append(f"Multiple orders: {len(self.pending_orders)}")

        return "; ".join(reasons) if reasons else "Risk threshold exceeded"


# Factory function for backward compatibility
async def create_trading_a2a_agent(
    model=None,
    is_debug: bool = False,
    checkpointer=None,
    risk_threshold: float = 0.15
) -> TradingA2AAgent:
    """
    트레이딩 A2A 에이전트를 생성하고 초기화합니다.

    Args:
        model: LLM 모델 (기본값: gpt-4.1-mini)
        is_debug: 디버그 모드 플래그
        checkpointer: 체크포인트 관리자
        risk_threshold: 승인이 필요한 VaR 임계값

    Returns:
        TradingA2AAgent: 초기화된 에이전트 인스턴스
    """
    agent = TradingA2AAgent(model, is_debug, checkpointer, risk_threshold)
    await agent.initialize()
    return agent
