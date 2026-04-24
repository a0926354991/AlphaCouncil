from google.adk.agents.llm_agent import Agent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.genai import types

from alpha_council.utils.master_runtime import DynamicMastersPanel, build_reports_context

from alpha_council.analysts import (
    technical_analyst,
    news_analyst,
    psychology_analyst,
    fundamental_analyst,
    chip_analyst,
)
from alpha_council.masters import (
    warren_buffett,
    ben_graham,
    charlie_munger,
    aswath_damodaran,
    bill_ackman,
    cathie_wood,
    michael_burry,
    peter_lynch,
    phil_fisher,
    mohnish_pabrai,
    stanley_druckenmiller,
    rakesh_jhunjhunwala,
    nassim_taleb,
)
from alpha_council.researchers import bull_researcher, bear_researcher
from alpha_council.risk import aggressive_debater, neutral_debater, conservative_debater
from alpha_council.trader import trader
from alpha_council.managers import research_manager
from alpha_council.master_selector import master_selector_agent
from guardrail.stock_code_guard import stock_code_guard_callback


# ---------------------------------------------------------------------------
# Pipeline-specific guard callbacks
# ---------------------------------------------------------------------------


def _skip_analyst_team(callback_context) -> types.Content | None:
    """Skip analyst_team when analysis_intent=False (chitchat) or awaiting master choice."""
    state = callback_context.state
    if state.get("analysis_intent") is False:
        return types.Content(parts=[])
    if state.get("awaiting_master_choice"):
        return types.Content(parts=[])
    return None


def _skip_downstream(callback_context) -> types.Content | None:
    """Skip downstream phases when awaiting master choice or no consolidated report yet."""
    state = callback_context.state
    if state.get("analysis_intent") is False:
        return types.Content(parts=[])
    if state.get("awaiting_master_choice"):
        return types.Content(parts=[])
    if not state.get("consolidated_masters_report"):
        return types.Content(parts=[])
    return None

# Phase 1 — 分析師團隊
# Skipped when analysis_intent=False (chitchat) or awaiting_master_choice=True (round 2).
analyst_team = ParallelAgent(
    name="analyst_team",
    sub_agents=[
        technical_analyst,
        news_analyst,
        psychology_analyst,
        fundamental_analyst,
        chip_analyst,
    ],
    before_agent_callback=_skip_analyst_team,
    description="並行執行技術、新聞、市場心理、籌碼面與基本面五位分析師，產出各自的分析報告。",
)

# Phase 1.5 — 大師選擇（使用者指定 3–7 位，或隨機 3 位）
# before_agent_callback=skip_if_no_analysis_intent already on master_selector_agent.
# Writes selected_masters: list[str] and awaiting_master_choice: bool to session state.

# Phase 2 — 13 位投資大師（僅執行已選中的大師）
# Skip logic handled inside DynamicMastersPanel._run_async_impl:
#   analysis_intent=False → skip; awaiting_master_choice=True → skip; selected empty → skip.
masters_panel = DynamicMastersPanel(
    name="masters_panel",
    sub_agents=[
        warren_buffett,
        ben_graham,
        charlie_munger,
        aswath_damodaran,
        bill_ackman,
        cathie_wood,
        michael_burry,
        peter_lynch,
        phil_fisher,
        mohnish_pabrai,
        stanley_druckenmiller,
        rakesh_jhunjhunwala,
        nassim_taleb,
    ],
    description="動態大師面板：只執行 state['selected_masters'] 中的大師，不產生未選中的 no-op 事件。",
)

# Phase 3 — 看多 / 看空研究員辯論（最多 2 輪）
research_debate = LoopAgent(
    name="research_debate",
    sub_agents=[bull_researcher, bear_researcher],
    max_iterations=2,
    before_agent_callback=_skip_downstream,
    description="看多研究員與看空研究員進行辯論，最多循環 2 輪，凝聚多空論點。",
)

# Phase 4 — 研究管理人裁決 → alpha_council.managers.research_manager
# Phase 4b — 交易員 → alpha_council.trader.trader

# Phase 5 — 風險辯論（最多 2 輪）
risk_debate = LoopAgent(
    name="risk_debate",
    sub_agents=[aggressive_debater, neutral_debater, conservative_debater],
    max_iterations=2,
    before_agent_callback=_skip_downstream,
    description="激進、中立、保守三位辯手對交易方案進行風險辯論，最多循環 2 輪。",
)

# Phase 6 — 投資組合管理人最終決策
def _portfolio_manager_instruction(ctx) -> str:
    context_block = build_reports_context(ctx.state, ["research_report?", "trader_plan?"])
    base = (
        "你是投資組合管理人，負責做出最終投資決策。\n\n"
        "根據研究管理人的裁決與交易員的執行計畫，輸出以下結構：\n"
        "1. **最終決策**：買入 / 持有 / 賣出（需與研究信號一致或說明偏差理由）\n"
        "2. **建議倉位比例**（相對總投資組合的百分比）\n"
        "3. **風險敞口控管**（最大可接受損失、停損設定）\n"
        "4. **退出策略**（目標價達成或停損觸發條件）\n"
    )
    if context_block:
        return (
            "【研究管理人裁決與交易員計畫】\n\n"
            f"{context_block}\n\n"
            "---\n\n"
            f"{base}"
        )
    return base

portfolio_manager = Agent(
    model="gemini-2.5-flash",
    name="portfolio_manager",
    description="整合所有分析與風險辯論，做出最終投資組合決策，包含倉位大小與風險控管措施。",
    before_agent_callback=_skip_downstream,
    instruction=_portfolio_manager_instruction,
)

# ---------------------------------------------------------------------------
# 主 Pipeline（SequentialAgent）
# 目前啟用順序：analyst_team → master_selector → masters_panel → research_debate → research_manager → trader → risk_debate → portfolio_manager
#
# 條件跳過由各 agent 的 before_agent_callback 負責；允許 skip events。
#
# Session state keys:
#   analyst_team         → news_report, technical_report, psychology_report, fundamentals_report, chip_report
#   master_selector      → selected_masters: list[str], awaiting_master_choice: bool
#   masters_panel        → {name}_report for each selected master
#                        + consolidated_masters_report
#   bull_researcher      → bull_argument  (每輪覆寫，第二輪已含對 bear 的回應)
#   bear_researcher      → bear_argument  (每輪覆寫，第二輪已含對 bull 的回應)
#   research_manager     → research_report
#   trader               → trader_plan

alpha_council_pipeline_agent = SequentialAgent(
    name="AlphaCouncilPipelineAgent",
    sub_agents=[
        analyst_team,
        master_selector_agent,
        masters_panel,
        research_debate,
        research_manager,
        trader,
        risk_debate,
        portfolio_manager,
    ],
    before_agent_callback=stock_code_guard_callback,
    description=(
        "AlphaCouncil 投資分析流水線（SequentialAgent）："
        "股票代號格式檢查 → 分析師團隊 → 大師選擇 → 大師觀點（含聚合）→ 研究辯論 → 研究裁決 → 交易員 → 風險辯論 → 投資組合管理人。"
        "各階段透過 before_agent_callback 條件跳過，允許 skip events。"
    ),
)

root_agent = alpha_council_pipeline_agent
