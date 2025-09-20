"""
A2A 통합이 적용된 분석(Analysis) 에이전트 V2.

이 모듈은 표준화된 A2A 인터페이스를 구현하여 종합적인 시장 분석을 수행하는
분석 에이전트를 제공합니다.

메모:
    - A2AOutput: 모든 A2A 에이전트에서 사용하는 타이핑된 딕셔너리로,
      스트리밍 이벤트와 최종 결과가 동일한 스키마(``status``,
      ``text_content``/``data_content``, ``final`` 플래그, ``metadata``)를 공유합니다.
    - 스트리밍 이벤트: LangGraph 는 ``on_llm_stream``, ``on_tool_start``,
      ``on_tool_end`` 등의 이벤트를 발생시킵니다. 의미 있는 부분만 A2AOutput으로
      변환하며, 나머지는 스트림을 깔끔하게 유지하기 위해 무시합니다.
    - 버퍼링: ``A2AStreamBuffer`` 는 잘게 쪼개진 토큰을 모아 간헐적으로 내보내어
      사용자가 읽기 쉬운 업데이트를 받도록 합니다.
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
    A2A 통합을 지원하는 분석 에이전트.

    이 에이전트는 다음의 4가지 차원에 대한 종합 분석을 수행합니다.
    - 기술적 분석 (Technical)
    - 기본적 분석 (Fundamental)
    - 감성 분석 (Sentiment)
    - 거시경제 분석 (Macro)

    결과는 범주형 신호(STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)를 제공합니다.

    설계 목표:
        - A2A 인터페이스로 스트리밍과 폴링을 통합
        - 내부 상태를 최소화하고 명시적으로 유지 (``analysis_dimensions``,
          ``final_signal``)
        - 기계/사람이 동시에 이해 가능한 출력 제공
    """

    def __init__(
        self,
        model=None,
        is_debug: bool = False,
        checkpointer=None
    ):
        """
        분석 A2A 에이전트 초기화.

        Args:
            model: 사용할 LLM 모델 (기본: gpt-4.1-mini)
            is_debug: 디버그 모드 여부
            checkpointer: 체크포인트 매니저 (기본: MemorySaver)
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
        """MCP 도구 로딩 및 그래프 생성 초기화.

        단계:
            1) MCP 를 통해 분석 도메인 도구 로딩 (``load_analysis_tools``)
            2) 도구 개수에 맞춘 시스템 프롬프트 구성
            3) 체크포인트가 포함된 ReAct 스타일 LangGraph 에이전트 생성

        Raises:
            RuntimeError: 도구/프롬프트/그래프 생성 중 오류 발생 시 래핑하여 전달
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
        """A2A 호환 입력/출력으로 에이전트를 실행합니다.

        Args:
            input_dict: ``{"messages": [...]}`` 형태의 페이로드 또는 구조화된
                분석 요청. 메시지는 LangChain 메시지 객체(예: ``HumanMessage``)여야 합니다.
            config: 선택적 실행 설정. 생략 시 고립된 실행을 위해 새로운 ``thread_id`` 를 생성합니다.

        Returns:
            A2AOutput: 그래프 실행 완료 시 최종 표준 출력

        Notes:
            - 예외는 ``format_error`` 를 통해 표준화된 에러 출력으로 변환됩니다.
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
        """LangGraph 스트리밍 이벤트를 ``A2AOutput`` 으로 변환합니다.

        Args:
            event: 이벤트 유형에 따라 ``event``, ``name`` 및 중첩된 ``data`` 등의 키를 포함하는 원시 이벤트 dict

        Returns:
            Optional[A2AOutput]: 내보낼 가치가 없는 이벤트(예: 버퍼링 중인 작은 토큰 조각)면 ``None``,
            그렇지 않으면 구조화된 업데이트를 반환합니다.

        지원되는 이벤트:
            - ``on_llm_stream``: 토큰 스트림 (버퍼링 후 플러시)
            - ``on_tool_start``/``on_tool_end``: 도구 라이프사이클 신호
            - 완료 이벤트 (``is_completion_event`` 참조)
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
        """LangGraph 실행 ``state`` 에서 최종 출력을 추출합니다.

        Args:
            state: 보통 실행 요약을 나타내는 마지막 ``AIMessage`` 가 포함된 ``messages`` 키를 갖는 상태 dict

        Returns:
            A2AOutput: 차원별 점수, 합성 점수, 최종 신호 등을 포함한 최종 메시지

        구현 메모:
            - 텍스트에 명시적 신호가 없으면 합성 점수로부터 신호를 도출하여
              비어 있는 추천을 반환하지 않도록 합니다.
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

            # Fallback: Try to get content from any message if AIMessage not found
            if not analysis_summary:
                for msg in messages:
                    if hasattr(msg, "content") and msg.content:
                        analysis_summary = str(msg.content)
                        logger.warning(f"No AIMessage found, using message content as fallback: {len(analysis_summary)} chars")
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

            # Ensure we have valid content for client processing
            if not analysis_summary:
                analysis_summary = "분석이 완료되었습니다."

            # A2AOutput의 text_content와 data_content가 모두 유효하도록 보장
            text_content = analysis_summary
            data_content = data_content or {}

            # text_content가 비어있을 경우 data_content에서 분석 내용을 추출하여 설정
            if not text_content or text_content == "분석이 완료되었습니다.":
                if data_content.get("result", {}).get("raw_analysis"):
                    text_content = data_content["result"]["raw_analysis"]
                else:
                    text_content = f"분석이 완료되었습니다. (신호: {self.final_signal}, 신뢰도: {confidence})"

            # Create final output with complete A2A standard compliance
            final_output = self.create_a2a_output(
                status="completed",
                text_content=text_content,
                data_content=data_content,
                final=True,
                metadata={
                    "execution_complete": True,
                    "final_signal": self.final_signal,
                    "confidence": confidence,
                    "analysis_dimensions": self.analysis_dimensions,  # 4차원 분석 결과 포함
                    "analysis_progress": self._get_analysis_progress()  # 분석 진행 상태 포함
                }
            )

            # A2AOutput 생성 성공 로그
            logger.info(f"A2AOutput 생성 완료: text_content={len(text_content)} chars, data_content keys={list(data_content.keys()) if data_content else 'None'}")
            logger.info(f"Final signal: {self.final_signal}, Confidence: {confidence}")
            logger.info(f"Analysis dimensions: {self.analysis_dimensions}")

            return final_output

        except Exception as e:
            logger.error(f"Error extracting final output: {e}", exc_info=True)
            logger.error(f"State keys: {list(state.keys()) if state else 'None'}")
            logger.error(f"Messages count: {len(state.get('messages', [])) if state else 'None'}")
            return self.format_error(e, context="extract_final_output")

    # Helper methods for analysis

    def _track_analysis_dimensions(self, content: str):
        """현재 처리 중인 분석 차원을 추적합니다.

        개선된 휴리스틱:
            - 각 차원별 키워드를 스캔하여 잠정적으로 "processing" 상태로 표시하고,
              이후 수치 점수가 수신되면 완료로 간주합니다.
            - 키워드 매칭 정확도를 높이기 위해 다양한 패턴 지원
            - 분석 심화도 추적을 위한 가중치 시스템 적용
        """
        if not content:
            return

        content_lower = content.lower()

        # 각 차원별 키워드 확장 (더 정확한 매칭을 위해)
        dimension_keywords = {
            "technical": [
                "기술적", "차트", "지표", "이동평균", "rsi", "macd", "bollinger",
                "stoch", "stochastics", "이평선", "기술분석", "차트분석",
                "지지선", "저항선", "패턴", "트렌드", "모멘텀", "과매수", "과매도"
            ],
            "fundamental": [
                "기본적", "재무", "실적", "per", "pbr", "roe", "roa", "eps", "bps",
                "기본분석", "재무분석", "기업가치", "밸류에이션", "배당", "배당률",
                "자기자본", "부채비율", "유동비율", "수익성", "성장성", "안정성"
            ],
            "sentiment": [
                "감성", "뉴스", "여론", "투자심리", "공포탐욕", "시장심리", "심리지수",
                "뉴스분석", "언론보도", "시장반응", "투자심리분석", "감정분석",
                "소셜미디어", "트위터", "레딧", "커뮤니티", "온라인반응"
            ],
            "macro": [
                "거시", "경제", "금리", "환율", "인플레이션", "gdp", "실업률", "경기",
                "거시경제", "경제지표", "통화정책", "재정정책", "경제성장률",
                "소비자물가", "생산자물가", "산업생산", "무역수지", "경상수지"
            ]
        }

        # 키워드 매칭 및 차원 상태 업데이트
        for dimension, keywords in dimension_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                if self.analysis_dimensions[dimension] is None:
                    self.analysis_dimensions[dimension] = "processing"
                    logger.debug(f"Analysis dimension '{dimension}' marked as processing")

    def _identify_analysis_dimension(self, tool_name: str) -> str:
        """도구 이름을 가장 가능성 높은 분석 차원으로 매핑합니다.

        신뢰 가능한 매핑이 없으면 ``"general"`` 로 대체합니다.
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
        """분석 진행 상태의 구조화된 스냅샷을 반환합니다."""
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
        """4차원 분석 결과를 종합하여 합성 점수와 신뢰도를 계산합니다.

        개선사항:
            - 각 차원의 가중치를 고려한 점수 계산
            - 분석 완료도에 따른 신뢰도 세분화
            - 데이터 부족 시의 적절한 기본값 처리
        """
        scores = []
        weights = {
            "technical": 0.25,      # 기술적 분석 25%
            "fundamental": 0.30,    # 기본적 분석 30% (가장 중요)
            "sentiment": 0.20,      # 감성 분석 20%
            "macro": 0.25          # 거시경제 분석 25%
        }

        # 각 차원의 점수 수집
        for dimension, value in self.analysis_dimensions.items():
            if isinstance(value, (int, float)):
                scores.append((value, dimension))
            elif value == "processing":
                # 진행 중인 분석은 중립 점수로 처리하되 가중치 감소
                scores.append((50, dimension))

        if not scores:
            logger.warning("No analysis scores available, returning neutral score")
            return 50.0, "LOW"

        # 가중치 적용된 합성 점수 계산
        weighted_sum = 0
        total_weight = 0

        for score, dimension in scores:
            weight = weights.get(dimension, 0.25)  # 기본 가중치 25%
            if dimension in self.analysis_dimensions and self.analysis_dimensions[dimension] == "processing":
                weight *= 0.5  # 진행 중인 차원은 가중치 절반으로 처리
            weighted_sum += score * weight
            total_weight += weight

        composite = weighted_sum / total_weight if total_weight > 0 else 50.0

        # 분석 완료도 기반 신뢰도 계산
        completed_dimensions = len([v for v in self.analysis_dimensions.values() if isinstance(v, (int, float))])
        processing_dimensions = len([v for v in self.analysis_dimensions.values() if v == "processing"])

        if completed_dimensions == 4:
            confidence = "VERY_HIGH"
        elif completed_dimensions == 3:
            confidence = "HIGH"
        elif completed_dimensions == 2:
            confidence = "MEDIUM"
        elif completed_dimensions == 1:
            confidence = "LOW"
        else:
            confidence = "VERY_LOW"

        # 처리 중인 차원이 있으면 신뢰도 한 단계 하향 조정
        if processing_dimensions > 0:
            confidence_levels = ["VERY_HIGH", "HIGH", "MEDIUM", "LOW", "VERY_LOW"]
            current_idx = confidence_levels.index(confidence)
            if current_idx < len(confidence_levels) - 1:
                confidence = confidence_levels[current_idx + 1]

        logger.info(f"Composite analysis: score={composite:.1f}, confidence={confidence}, "
                   f"completed={completed_dimensions}, processing={processing_dimensions}")

        return round(composite, 1), confidence

    def _determine_signal(self, score: float) -> str:
        """합성 점수를 범주형 추천 신호로 매핑합니다."""
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
        """비정형 텍스트에서 추천 신호를 추출합니다.

        영어와 한국어 패턴을 모두 지원하며, 우선순위 기반 매칭을 수행합니다.

        우선순위:
        1. 강력매수/강력매도 (STRONG_BUY/STRONG_SELL)
        2. 매수/매도 (BUY/SELL)
        3. 중립/보유 (HOLD)
        4. 영문 패턴 매칭
        """
        if not text:
            return None

        text_lower = text.lower()
        text_upper = text.upper()

        # 우선순위 1: 강력한 신호 패턴 (정확한 매칭 우선)
        strong_patterns = {
            "STRONG_BUY": [
                "강력매수", "적극매수", "강력 매수", "적극 매수",
                "매우매수", "매우 매수", "strongly buy", "definitely buy"
            ],
            "STRONG_SELL": [
                "강력매도", "적극매도", "강력 매도", "적극 매도",
                "매우매도", "매우 매도", "strongly sell", "definitely sell"
            ]
        }

        for signal, patterns in strong_patterns.items():
            for pattern in patterns:
                if pattern in text or pattern in text_lower:
                    return signal

        # 우선순위 2: 일반 매수/매도 패턴
        if any(pattern in text_lower for pattern in ["매수", "buy", "구매"]):
            return "BUY"
        elif any(pattern in text_lower for pattern in ["매도", "sell", "판매"]):
            return "SELL"

        # 우선순위 3: 중립/보유 패턴
        if any(pattern in text_lower for pattern in ["중립", "보유", "hold", "유지", "관망"]):
            return "HOLD"

        # 우선순위 4: 영문 패턴 매칭 (덜 구체적)
        signals = ["STRONG_BUY", "STRONG_SELL", "BUY", "SELL", "HOLD"]
        for signal in signals:
            if signal in text_upper:
                return signal

        return None


# Factory function for backward compatibility
async def create_analysis_a2a_agent(
    model=None,
    is_debug: bool = False,
    checkpointer=None
) -> AnalysisA2AAgent:
    """
    분석 A2A 에이전트를 생성하고 초기화합니다.

    Args:
        model: LLM 모델 (기본: gpt-4.1-mini)
        is_debug: 디버그 모드 플래그
        checkpointer: 체크포인트 매니저

    Returns:
        AnalysisA2AAgent: 초기화된 에이전트 인스턴스
    """
    agent = AnalysisA2AAgent(model, is_debug, checkpointer)
    await agent.initialize()
    return agent
