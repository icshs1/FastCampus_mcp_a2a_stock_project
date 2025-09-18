"""
Analysis Agent with A2A Integration.

This module provides an Analysis agent that implements the standardized
A2A interface for comprehensive market analysis.

Beginner notes:
    - A2AOutput: A typed dict used across all A2A agents so that streaming
      events and final results share a consistent schema (``status``,
      ``text_content``/``data_content``, ``final`` flags, and ``metadata``).
    - Streaming events: LangGraph emits events such as ``on_llm_stream``,
      ``on_tool_start``, ``on_tool_end``. We translate only meaningful parts
      to A2AOutput. Everything else is ignored to keep the stream clean.
    - Buffering: ``A2AStreamBuffer`` aggregates tiny token chunks and flushes
      intermittently so clients receive readable updates instead of raw tokens.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import pytz
import structlog
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.lg_agents.base.a2a_interface import A2AOutput, A2AStreamBuffer, BaseA2AAgent
from src.lg_agents.base.base_graph_agent import BaseGraphAgent
from src.lg_agents.base.mcp_config import load_analysis_tools
from src.lg_agents.prompts import get_prompt
from src.lg_agents.util import load_env_file

logger = structlog.get_logger(__name__)

load_env_file()


class AnalysisA2AAgent(BaseA2AAgent, BaseGraphAgent):
    """
    Analysis Agent with A2A integration support.

    This agent performs comprehensive 4-dimensional analysis:
    - Technical Analysis
    - Fundamental Analysis
    - Sentiment Analysis
    - Macro Economic Analysis

    Provides category-based signals: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL

    Design goals:
        - Unify streaming and polling via A2A interface
        - Keep internal state minimal and explicit (``analysis_dimensions``,
          ``final_signal``)
        - Make outputs machine- and human-consumable simultaneously
    """

    def __init__(
        self,
        model=None,
        is_debug: bool = False,
        checkpointer=None
    ):
        """
        Initialize Analysis A2A Agent.

        Args:
            model: LLM model (default: gpt-4o-mini)
            is_debug: Debug mode flag
            checkpointer: Checkpoint manager (default: MemorySaver)
        """
        BaseA2AAgent.__init__(self)

        # Initialize the model
        self.model = model or init_chat_model(
            model="gpt-4.1-mini",
            temperature=0,
            model_provider="openai"
        )
        self.checkpointer = checkpointer or MemorySaver()

        # Initialize BaseGraphAgent with required parameters
        BaseGraphAgent.__init__(
            self,
            model=self.model,
            checkpointer=self.checkpointer,
            is_debug=is_debug,
            lazy_init=True,  # Use lazy initialization for A2A agents
            agent_name="AnalysisA2AAgent"
        )

        self.tools = []

        # Stream buffer for managing LLM output
        self.stream_buffer = A2AStreamBuffer(max_size=100)

        # Track analysis progress
        self.analyzed_symbols = set()
        self.analysis_dimensions = {
            "technical": None,
            "fundamental": None,
            "sentiment": None,
            "macro": None
        }
        self.final_signal: Optional[str] = None

    async def initialize(self):
        """Initialize the agent with MCP tools and create the graph.

        Steps:
            1) Load domain tools via MCP (``load_analysis_tools``)
            2) Build the system prompt tailored by tool count
            3) Create a ReAct-style LangGraph agent with checkpointing

        Raises:
            RuntimeError: When tool loading, prompt building, or graph creation
                fails; the original exception is attached as context.
        """
        try:
            # Load MCP tools
            self.tools = await load_analysis_tools()
            logger.info(f"Loaded {len(self.tools)} MCP tools for Analysis")

            # Get system prompt
            system_prompt = get_prompt("analysis", "system", tool_count=len(self.tools))

            # Create the reactive agent graph
            config = RunnableConfig(recursion_limit=10)
            self.graph = create_react_agent(
                model=self.model,
                tools=self.tools,
                prompt=system_prompt,
                checkpointer=self.checkpointer,
                name="AnalysisAgent",
                debug=self.is_debug,
                context_schema=config
            )

            logger.info("Analysis A2A Agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Analysis Agent: {e}")
            raise RuntimeError(f"Analysis Agent initialization failed: {e}") from e

    async def execute_for_a2a(
        self,
        input_dict: dict[str, Any],
        config: Optional[dict[str, Any]] = None
    ) -> A2AOutput:
        """Execute the agent with A2A-compatible input and output.

        Args:
            input_dict: Either a ``{"messages": [...]}`` style payload or a
                structured analysis request. Messages should be LangChain
                Message objects (e.g., ``HumanMessage``).
            config: Optional execution config. If omitted, a fresh
                ``thread_id`` is generated to isolate this run.

        Returns:
            A2AOutput: Final standardized output when the graph completes.

        Notes:
            - Any exceptions are converted to a standardized error A2AOutput
              via ``format_error`` to keep client behavior uniform.
        """
        if not self.graph:
            await self.initialize()

        try:
            # Reset tracking variables
            self.analyzed_symbols.clear()
            self.analysis_dimensions = {
                "technical": None,
                "fundamental": None,
                "sentiment": None,
                "macro": None
            }
            self.final_signal = None

            # Execute the graph
            result = await self.graph.ainvoke(
                input_dict,
                config=config or {"configurable": {"thread_id": str(uuid4())}},
            )

            logger.info(f"[AnalysisA2AAgent] result: {result}")

            # Extract final output
            return self.extract_final_output(result)

        except Exception as e:
            return self.format_error(e, context="execute_for_a2a")

    def format_stream_event(
        self,
        event: dict[str, Any]
    ) -> Optional[A2AOutput]:
        """Convert a LangGraph streaming event to an ``A2AOutput``.

        Args:
            event: Raw event dict (expects keys like ``event``, ``name``,
                and nested ``data`` depending on the event type).

        Returns:
            Optional[A2AOutput]: ``None`` when the event does not warrant an
            outbound message (e.g., small token chunk buffered), otherwise a
            structured update.

        Supported events:
            - ``on_llm_stream``: token chunks; buffered and flushed
            - ``on_tool_start``/``on_tool_end``: tool lifecycle signals
            - completion events (see ``is_completion_event``)
        """
        event_type = event.get("event", "")

        # Handle LLM streaming
        if event_type == "on_llm_stream":
            content = self.extract_llm_content(event)
            if content:
                # Track analysis dimensions mentioned in content
                self._track_analysis_dimensions(content)

                if self.stream_buffer.add(content):
                    # Buffer is full, flush it
                    return self.create_a2a_output(
                        status="working",
                        text_content=self.stream_buffer.flush(),
                        stream_event=True,
                        metadata={
                            "event_type": "llm_stream",
                            "analysis_progress": self._get_analysis_progress()
                        }
                    )

        # Handle tool execution events
        elif event_type == "on_tool_start":
            tool_name = event.get("name", "unknown")
            dimension = self._identify_analysis_dimension(tool_name)

            return self.create_a2a_output(
                status="working",
                text_content=f" 분석 진행 중: {dimension} 분석",
                stream_event=True,
                metadata={
                    "event_type": "tool_start",
                    "tool_name": tool_name,
                    "analysis_dimension": dimension
                }
            )

        # Handle tool completion with analysis results
        elif event_type == "on_tool_end":
            tool_output = event.get("data", {}).get("output", {})
            tool_name = event.get("name", "unknown")
            dimension = self._identify_analysis_dimension(tool_name)

            if tool_output and isinstance(tool_output, dict):
                # Update dimension scores
                if dimension and "score" in tool_output:
                    self.analysis_dimensions[dimension] = tool_output.get("score")

                return self.create_a2a_output(
                    status="working",
                    data_content={
                        "analysis_update": {
                            "dimension": dimension,
                            "result": tool_output
                        },
                        "current_progress": self._get_analysis_progress()
                    },
                    stream_event=True,
                    metadata={"event_type": "analysis_update"}
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
        state: dict[str, Any]
    ) -> A2AOutput:
        """Extract final output from the LangGraph run ``state``.

        Args:
            state: Run state dict that usually contains ``messages`` with the
                last ``AIMessage`` representing the run summary.

        Returns:
            A2AOutput: Final message including structured analysis fields
            (per-dimension scores, composite score, and signal).

        Implementation notes:
            - If an explicit signal is not present in text, we derive it from
              the composite score to avoid returning an empty recommendation.
        """
        try:
            # Extract messages from state
            messages = state.get("messages", [])

            # Get the last AI message as analysis summary
            analysis_summary = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
                    analysis_summary = msg.content
                    # Extract signal from summary
                    self.final_signal = self._extract_signal_from_text(analysis_summary)
                    break

            # Calculate composite score and confidence
            composite_score, confidence = self._calculate_composite_analysis()

            # Determine final signal if not extracted
            if not self.final_signal:
                self.final_signal = self._determine_signal(composite_score)

            # Prepare structured data
            data_content = {
                "success": True,
                "result": {
                    "raw_analysis": analysis_summary,
                    "symbols_analyzed": list(self.analyzed_symbols),
                    "analysis_signal": self.final_signal,
                    "technical_score": self.analysis_dimensions.get("technical", 0),
                    "fundamental_score": self.analysis_dimensions.get("fundamental", 0),
                    "sentiment_score": self.analysis_dimensions.get("sentiment", 0),
                    "macro_score": self.analysis_dimensions.get("macro", 0),
                    "composite_score": composite_score,
                    "confidence_level": confidence,
                    "timestamp": datetime.now(pytz.UTC).isoformat()
                },
                "agent_type": "AnalysisA2AAgent",
                "workflow_status": "completed"
            }

            # Create final output
            return self.create_a2a_output(
                status="completed",
                text_content=analysis_summary or "분석이 완료되었습니다.",
                data_content=data_content,
                final=True,
                metadata={
                    "execution_complete": True,
                    "final_signal": self.final_signal,
                    "confidence": confidence
                }
            )

        except Exception as e:
            logger.error(f"Error extracting final output: {e}")
            return self.format_error(e, context="extract_final_output")

    # Helper methods for analysis

    def _track_analysis_dimensions(self, content: str):
        """Track which analysis dimensions are being processed.

        Heuristics:
            - Keyword scanning per dimension marks a dimension as
              "processing" until a numeric score is later received.
        """
        content_lower = content.lower()

        dimension_keywords = {
            "technical": ["기술적", "차트", "지표", "이동평균", "rsi", "macd"],
            "fundamental": ["기본적", "재무", "실적", "per", "pbr", "roe"],
            "sentiment": ["감성", "뉴스", "여론", "투자심리", "공포탐욕"],
            "macro": ["거시", "경제", "금리", "환율", "인플레이션"]
        }

        for dimension, keywords in dimension_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                if self.analysis_dimensions[dimension] is None:
                    self.analysis_dimensions[dimension] = "processing"

    def _identify_analysis_dimension(self, tool_name: str) -> str:
        """Map a tool name to the most likely analysis dimension.

        Falls back to ``"general"`` when no reliable mapping exists.
        """
        tool_lower = tool_name.lower()

        if any(x in tool_lower for x in ["technical", "chart", "indicator"]):
            return "technical"
        elif any(x in tool_lower for x in ["fundamental", "financial", "earnings"]):
            return "fundamental"
        elif any(x in tool_lower for x in ["sentiment", "news", "social"]):
            return "sentiment"
        elif any(x in tool_lower for x in ["macro", "economic", "market"]):
            return "macro"

        return "general"

    def _get_analysis_progress(self) -> dict[str, Any]:
        """Return a structured snapshot of analysis progress."""
        completed = sum(
            1 for v in self.analysis_dimensions.values()
            if v is not None and v != "processing"
        )

        return {
            "dimensions_completed": completed,
            "dimensions_total": 4,
            "completion_percentage": (completed / 4) * 100,
            "dimensions_status": self.analysis_dimensions.copy()
        }

    def _calculate_composite_analysis(self) -> tuple[float, str]:
        """Calculate a composite score and qualitative confidence level."""
        scores = []

        for _dimension, value in self.analysis_dimensions.items():
            if isinstance(value, (int, float)):
                scores.append(value)
            elif value == "processing":
                scores.append(50)  # Neutral for incomplete

        if not scores:
            return 50.0, "LOW"

        composite = sum(scores) / len(scores)

        # Determine confidence based on completion
        completed_count = len([v for v in self.analysis_dimensions.values() if isinstance(v, (int, float))])

        if completed_count >= 4:
            confidence = "HIGH"
        elif completed_count >= 3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return composite, confidence

    def _determine_signal(self, score: float) -> str:
        """Map a composite score to a categorical recommendation signal."""
        if score >= 80:
            return "STRONG_BUY"
        elif score >= 60:
            return "BUY"
        elif score >= 40:
            return "HOLD"
        elif score >= 20:
            return "SELL"
        else:
            return "STRONG_SELL"

    def _extract_signal_from_text(self, text: str) -> Optional[str]:
        """Extract a recommendation signal from unstructured text.

        Supports both English (e.g., ``BUY``) and Korean (예: ``매수``).
        """
        text_upper = text.upper()

        signals = ["STRONG_BUY", "STRONG_SELL", "BUY", "SELL", "HOLD"]

        for signal in signals:
            if signal in text_upper:
                return signal

        # Check Korean equivalents
        if "강력매수" in text or "적극매수" in text:
            return "STRONG_BUY"
        elif "매수" in text:
            return "BUY"
        elif "강력매도" in text or "적극매도" in text:
            return "STRONG_SELL"
        elif "매도" in text:
            return "SELL"
        elif "중립" in text or "보유" in text:
            return "HOLD"

        return None


# Factory function for backward compatibility
async def create_analysis_a2a_agent(
    model=None,
    is_debug: bool = False,
    checkpointer=None
) -> AnalysisA2AAgent:
    """
    Create and initialize an Analysis A2A Agent.

    Args:
        model: LLM model (default: gpt-4o-mini)
        is_debug: Debug mode flag
        checkpointer: Checkpoint manager

    Returns:
        AnalysisA2AAgent: Initialized agent instance
    """
    agent = AnalysisA2AAgent(model, is_debug, checkpointer)
    await agent.initialize()
    return agent
