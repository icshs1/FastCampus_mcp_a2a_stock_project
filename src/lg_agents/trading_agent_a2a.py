"""
Trading Agent with A2A Integration.

This module provides a Trading agent that implements the standardized
A2A interface with Human-in-the-Loop approval for risk management.

Beginner notes:
    - Approval flow: High-risk operations return ``status='input_required'``
      with an approval request payload. Clients must respond with
      "approve/승인" or "reject/거부" as a follow-up message.
    - Risk threshold: ``risk_threshold`` gates when approval is required.
      Additional safeguards include large order detection and number of
      simultaneous pending orders.
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
    Trading Agent with A2A integration and Human-in-the-Loop support.

    This agent handles:
    - Order execution and management
    - Risk assessment (VaR, Sharpe ratio)
    - Portfolio optimization
    - Human approval workflow for high-risk trades
    """

    def __init__(
        self,
        model=None,
        is_debug: bool = False,
        checkpointer=None,
        risk_threshold: float = 0.15  # 15% VaR threshold
    ):
        """
        Initialize Trading A2A Agent.

        Args:
            model: LLM model (default: gpt-4o-mini)
            is_debug: Debug mode flag
            checkpointer: Checkpoint manager (default: MemorySaver)
            risk_threshold: VaR threshold for requiring approval
        """
        BaseA2AAgent.__init__(self)

        # Initialize the model
        self.model = model or init_chat_model(
            model="gpt-4o-mini",
            temperature=0,
            model_provider="openai"
        )
        self.checkpointer = checkpointer or MemorySaver()
        self.risk_threshold = risk_threshold

        # Initialize BaseGraphAgent with required parameters
        BaseGraphAgent.__init__(
            self,
            model=self.model,
            checkpointer=self.checkpointer,
            is_debug=is_debug,
            lazy_init=True,  # Use lazy initialization for A2A agents
            agent_name="TradingA2AAgent"
        )

        self.tools = []

        # Stream buffer for managing LLM output
        self.stream_buffer = A2AStreamBuffer(max_size=100)

        # Track trading state
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
        """Initialize the agent with MCP tools and create the graph.

        Raises:
            RuntimeError: If tools cannot be loaded or the graph fails to build.
        """
        try:
            # Load MCP tools
            self.tools = await load_trading_tools()
            logger.info(f" Loaded {len(self.tools)} MCP tools for Trading")

            # Get system prompt
            system_prompt = get_prompt("trading", "system", tool_count=len(self.tools))

            # Create the reactive agent graph
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
        Execute the agent with A2A-compatible input and output.

        Includes Human-in-the-Loop approval for high-risk trades.

        Args:
            input_dict: Trading request (often ``{"messages": [...]}``).
            config: Optional execution config; default ``thread_id`` is used
                when omitted.

        Returns:
            A2AOutput: Standardized output for A2A processing
        """
        if not self.graph:
            await self.initialize()

        try:
            # Reset tracking variables
            self.pending_orders.clear()
            self.executed_trades.clear()
            self.requires_approval = False
            self.approval_status = "PENDING"

            # Check if this is a resume operation (Human response)
            if self._is_approval_response(input_dict):
                return await self._handle_approval_response(input_dict)

            # Execute the graph
            result = await self.graph.ainvoke(
                input_dict,
                config=config or {"configurable": {"thread_id": str(uuid4())}}
            )

            # Check if approval is needed
            if self._requires_human_approval():
                return self._create_approval_request()

            # Extract final output
            return self.extract_final_output(result)

        except Exception as e:
            return self.format_error(e, context="execute_for_a2a")

    def format_stream_event(
        self,
        event: Dict[str, Any]
    ) -> Optional[A2AOutput]:
        """Translate LangGraph event to A2A streaming update.

        Emits human-friendly text for order/risk/portfolio phases.
        """
        event_type = event.get("event", "")

        # Handle LLM streaming
        if event_type == "on_llm_stream":
            content = self.extract_llm_content(event)
            if content and self.stream_buffer.add(content):
                # Buffer is full, flush it
                return self.create_a2a_output(
                    status="working",
                    text_content=self.stream_buffer.flush(),
                    stream_event=True,
                    metadata={"event_type": "llm_stream"}
                )

        # Handle tool execution events
        elif event_type == "on_tool_start":
            tool_name = event.get("name", "unknown")

            # Track different tool types
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

        # Handle tool completion with results
        elif event_type == "on_tool_end":
            tool_output = event.get("data", {}).get("output", {})
            tool_name = event.get("name", "unknown")

            # Process risk assessment results
            if "risk" in tool_name.lower() or "var" in tool_name.lower():
                if isinstance(tool_output, dict):
                    var_ratio = tool_output.get("var_ratio", 0)
                    self.risk_metrics["var_ratio"] = var_ratio

                    # Check if approval is needed
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

            # Process order results
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

        # Handle completion events
        elif self.is_completion_event(event):
            # Flush any remaining buffer content
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
        """Create the final ``A2AOutput`` including risk and approval fields."""
        try:
            # Extract messages from state
            messages = state.get("messages", [])

            # Get the last AI message as trading summary
            trading_summary = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
                    trading_summary = msg.content
                    break

            # Determine trading signal
            trading_signal = self._extract_trading_action(trading_summary)

            # Calculate total order amount
            total_amount = sum(
                order.get("amount", 0) for order in self.pending_orders
            )

            # Prepare structured data
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

            # Determine final status
            status = "input_required" if self.requires_approval else "completed"

            # Create final output
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
        """Return True when a human must approve before proceeding."""
        # Check VaR threshold
        if self.risk_metrics["var_ratio"] > self.risk_threshold:
            return True

        # Check for large orders
        total_amount = sum(order.get("amount", 0) for order in self.pending_orders)
        if total_amount > 10000000:  # 10M KRW threshold
            return True

        # Check for multiple orders
        if len(self.pending_orders) > 5:
            return True

        return False

    def _create_approval_request(self) -> A2AOutput:
        """Build an approval request output for human review."""
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
        """Create the human-facing message content for approval UI."""
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
        """Detect whether the latest message contains an approval decision."""
        messages = input_dict.get("messages", [])
        if not messages:
            return False

        last_message = messages[-1]
        if hasattr(last_message, "content"):
            content = last_message.content.lower()
            return any(word in content for word in ["approve", "승인", "reject", "거부", "cancel", "취소"])

        return False

    async def _handle_approval_response(self, input_dict: Dict[str, Any]) -> A2AOutput:
        """Apply a user's approval/denial decision and finalize output."""
        messages = input_dict.get("messages", [])
        response = messages[-1].content.lower() if messages else ""

        if "approve" in response or "승인" in response:
            self.approval_status = "APPROVED"
            # Execute pending orders
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

    # Helper methods

    def _extract_symbols_from_orders(self) -> list[str]:
        """Return the unique set of symbols referenced by pending/executed orders."""
        symbols = set()
        for order in self.pending_orders + self.executed_trades:
            if "symbol" in order:
                symbols.add(order["symbol"])
        return list(symbols)

    def _extract_trading_action(self, text: str) -> str:
        """Parse BUY/SELL/HOLD intent from unstructured text."""
        text_lower = text.lower()

        if "매수" in text or "buy" in text_lower:
            return "BUY"
        elif "매도" in text or "sell" in text_lower:
            return "SELL"
        elif "보유" in text or "hold" in text_lower:
            return "HOLD"

        return "PENDING"

    def _determine_risk_level(self) -> str:
        """Map VaR ratio to qualitative risk tiers."""
        var_ratio = self.risk_metrics["var_ratio"]

        if var_ratio > 0.20:
            return "HIGH"
        elif var_ratio > 0.10:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_approval_reason(self) -> str:
        """Explain why approval is required for the current state."""
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
    Create and initialize a Trading A2A Agent.

    Args:
        model: LLM model (default: gpt-4o-mini)
        is_debug: Debug mode flag
        checkpointer: Checkpoint manager
        risk_threshold: VaR threshold for approval

    Returns:
        TradingA2AAgent: Initialized agent instance
    """
    agent = TradingA2AAgent(model, is_debug, checkpointer, risk_threshold)
    await agent.initialize()
    return agent
