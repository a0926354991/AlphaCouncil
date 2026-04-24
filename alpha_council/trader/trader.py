from google.adk.agents.llm_agent import Agent
from google.genai import types

from alpha_council.utils.master_runtime import build_reports_context


def _skip_downstream(callback_context) -> types.Content | None:
    state = callback_context.state
    if state.get("analysis_intent") is False:
        return types.Content(parts=[])
    if state.get("awaiting_master_choice"):
        return types.Content(parts=[])
    if not state.get("consolidated_masters_report"):
        return types.Content(parts=[])
    return None


def _trader_instruction(ctx) -> str:
    state = ctx.state
    research_block = build_reports_context(state, ["research_report?"])

    base = (
        "你是交易員，負責將研究管理人的裁決轉化為可立即執行的交易指令。\n\n"
        "【輸出結構 — 必須依序包含以下五個部分】\n\n"
        "1. **交易標的**：確認 ticker 與公司名稱。\n\n"
        "2. **交易方向**：買入 / 持有 / 賣出（需與研究管理人裁決一致，若有偏差須說明原因）。\n\n"
        "3. **執行計畫**：\n"
        "   - 建議進場價格區間或觸發條件（如突破、回測支撐）\n"
        "   - 建議倉位規模（佔總投資組合的概略比例）\n"
        "   - 分批進場策略（若適用）\n\n"
        "4. **風險控管**：\n"
        "   - 停損價格或條件（需量化，如「跌破 XX 元」）\n"
        "   - 最大可接受損失（佔倉位 % 或絕對金額估算）\n\n"
        "5. **出場策略**：\n"
        "   - 目標價或獲利了結條件\n"
        "   - 若基本面或技術面惡化，提前出場的觸發條件\n\n"
        "【格式要求】\n"
        "回應最後一行必須為：\n"
        "FINAL TRANSACTION PROPOSAL: **買入** / **持有** / **賣出**（擇一）"
    )

    if research_block:
        return (
            "【研究管理人裁決結論 — 交易指令依據】\n\n"
            f"{research_block}\n\n"
            "---\n\n"
            f"{base}"
        )
    return base


trader = Agent(
    model="gemini-2.5-flash",
    name="trader",
    description="依據研究管理人的結論，擬定具體可執行的交易指令（方向、倉位、停損、出場）。",
    before_agent_callback=_skip_downstream,
    instruction=_trader_instruction,
    output_key="trader_plan",
)
