"""Mohnish Pabrai — Dhandho: heads I win, tails I don't lose much.

Pabrai's edge is *asymmetric risk-reward*: he copies known winners (clone
investing) and pays only when the downside is bounded by tangible value
plus a wide margin of safety. Versus Buffett he is stricter on price and
more flexible on quality.

  Owner-earnings yield (3)  > 8% (3) | > 5% (1)
  Margin of safety (3)      MoS > 50% (3) | > 30% (2) | > 10% (1)
                            intrinsic = OE × 10
  Balance sheet (2)         D/E < 0.3 (2) | < 0.5 (1)
  Predictability (2)        FCF positive 3/3 (2) | 2/3 (1)
  Capital allocation (2)    Net buybacks (1) | Insider net buying (1)

Total cap 12. Note "Dhandho" implicitly demands a *simple* business — not
something we can score quantitatively from yfinance, so the persona prompt
carries that judgement instead of the deterministic side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from alpha_council.masters._scoring.common import (
    ScoreLine,
    compute_owner_earnings,
    format_scorecard,
    latest,
    line_item_series,
    money,
    num,
    pct,
)
from alpha_council.providers.base import FinancialMetrics, InsiderTrade, LineItem


@dataclass(frozen=True)
class PabraiScore:
    owner_earnings_yield_lines: list[ScoreLine] = field(default_factory=list)
    margin_of_safety_lines: list[ScoreLine] = field(default_factory=list)
    balance_sheet: list[ScoreLine] = field(default_factory=list)
    predictability: list[ScoreLine] = field(default_factory=list)
    capital_allocation: list[ScoreLine] = field(default_factory=list)
    intrinsic_value: float | None = None
    market_cap: float | None = None
    margin_of_safety: float | None = None
    owner_earnings_yield: float | None = None

    @property
    def total(self) -> float:
        return sum(l.score for g in self._groups for l in g)

    @property
    def total_max(self) -> float:
        return sum(l.max_score for g in self._groups for l in g)

    @property
    def _groups(self) -> tuple[list[ScoreLine], ...]:
        return (
            self.owner_earnings_yield_lines,
            self.margin_of_safety_lines,
            self.balance_sheet,
            self.predictability,
            self.capital_allocation,
        )


def _valuation(latest_fm: FinancialMetrics | None,
                line_items: Iterable[LineItem],
                market_cap: float | None) -> tuple[
                    list[ScoreLine], list[ScoreLine],
                    float | None, float | None, float | None]:
    """Returns (oey_lines, mos_lines, intrinsic, oe_yield, mos)."""
    owner_earnings, derivation = compute_owner_earnings(latest_fm, line_items)
    if owner_earnings is None or market_cap is None or market_cap == 0:
        return (
            [ScoreLine("OE yield > 8%", 0.0, 3.0,
                       f"{derivation}; market_cap={money(market_cap)}")],
            [ScoreLine("Margin of safety > 50%", 0.0, 3.0, "intrinsic unavailable")],
            None, None, None,
        )
    oe_yield = owner_earnings / market_cap
    intrinsic = owner_earnings * 10
    mos = (intrinsic - market_cap) / intrinsic if intrinsic > 0 else None

    oey_score = 3.0 if oe_yield > 0.08 else (1.0 if oe_yield > 0.05 else 0.0)
    oey_line = ScoreLine("OE yield > 8%", oey_score, 3.0,
                         f"OE yield={pct(oe_yield)} ({derivation}); intrinsic≈{money(intrinsic)}")

    if mos is None:
        mos_score = 0.0
        mos_detail = "MoS undefined"
    elif mos > 0.50:
        mos_score = 3.0
        mos_detail = f"MoS={pct(mos)} (>50% — Pabrai strong-buy zone)"
    elif mos > 0.30:
        mos_score = 2.0
        mos_detail = f"MoS={pct(mos)} (>30%)"
    elif mos > 0.10:
        mos_score = 1.0
        mos_detail = f"MoS={pct(mos)} (>10% — thin)"
    else:
        mos_score = 0.0
        mos_detail = f"MoS={pct(mos)} (insufficient)"
    return [oey_line], [ScoreLine("Margin of safety > 50%", mos_score, 3.0, mos_detail)], intrinsic, oe_yield, mos


def _balance_sheet(latest_fm: FinancialMetrics | None) -> list[ScoreLine]:
    if latest_fm is None or latest_fm.debt_to_equity is None:
        return [ScoreLine("D/E < 0.3", 0.0, 2.0, "D/E n/a")]
    de = latest_fm.debt_to_equity
    score = 2.0 if de < 0.3 else (1.0 if de < 0.5 else 0.0)
    return [ScoreLine("D/E < 0.3", score, 2.0, f"D/E={num(de)}")]


def _predictability(line_items: Iterable[LineItem]) -> list[ScoreLine]:
    fcf_series = list(line_item_series(list(line_items), "free_cash_flow"))[:3]
    if not fcf_series:
        return [ScoreLine("FCF positive (3y)", 0.0, 2.0, "FCF history unavailable")]
    positives = sum(1 for li in fcf_series if (li.value or 0) > 0)
    score = 2.0 if positives >= 3 else (1.0 if positives >= 2 else 0.0)
    return [ScoreLine("FCF positive (3y)", score, 2.0,
                       f"{positives}/{len(fcf_series)} positive years")]


def _capital_allocation(line_items: Iterable[LineItem],
                         insider_trades: list[InsiderTrade]) -> list[ScoreLine]:
    lines = []
    shares = list(line_item_series(list(line_items), "shares_outstanding"))
    if len(shares) >= 2 and shares[0].value is not None and shares[-1].value is not None:
        delta = shares[0].value - shares[-1].value
        lines.append(ScoreLine(
            "Net buybacks",
            1.0 if delta < 0 else 0.0,
            1.0,
            f"shares {money(shares[-1].value)} → {money(shares[0].value)}",
        ))
    else:
        lines.append(ScoreLine("Net buybacks", 0.0, 1.0, "shares history n/a"))

    if not insider_trades:
        lines.append(ScoreLine("Insider net buying", 0.0, 1.0, "no insider data"))
    else:
        buy = sum(t.shares or 0 for t in insider_trades if t.transaction_type == "buy")
        sell = sum(abs(t.shares or 0) for t in insider_trades
                   if t.transaction_type in ("sell", "planned_sell"))
        net = buy - sell
        lines.append(ScoreLine(
            "Insider net buying",
            1.0 if net > 0 else 0.0,
            1.0,
            f"net Δ={money(net)} shares across {len(insider_trades)} rows",
        ))
    return lines


def score(state) -> PabraiScore:
    metrics: list[FinancialMetrics] = state.get("shared_data:financial_metrics") or []
    line_items: list[LineItem] = state.get("shared_data:line_items") or []
    insider: list[InsiderTrade] = state.get("shared_data:insider_trades") or []
    market_cap = state.get("shared_data:market_cap")
    latest_fm = latest(metrics)
    oey_lines, mos_lines, intrinsic, oe_yield, mos = _valuation(latest_fm, line_items, market_cap)
    return PabraiScore(
        owner_earnings_yield_lines=oey_lines,
        margin_of_safety_lines=mos_lines,
        balance_sheet=_balance_sheet(latest_fm),
        predictability=_predictability(line_items),
        capital_allocation=_capital_allocation(line_items, insider),
        intrinsic_value=intrinsic,
        market_cap=market_cap,
        margin_of_safety=mos,
        owner_earnings_yield=oe_yield,
    )


def format_block(s: PabraiScore) -> str:
    sections = [
        format_scorecard("Owner Earnings Yield", s.owner_earnings_yield_lines),
        format_scorecard("Margin of Safety", s.margin_of_safety_lines),
        format_scorecard("Balance Sheet", s.balance_sheet),
        format_scorecard("Predictability (FCF)", s.predictability),
        format_scorecard("Capital Allocation", s.capital_allocation),
    ]
    summary = (
        f"### Summary\n"
        f"  total: {s.total:.1f} / {s.total_max:.1f}\n"
        f"  intrinsic_value: {money(s.intrinsic_value)}\n"
        f"  market_cap: {money(s.market_cap)}\n"
        f"  margin_of_safety: {pct(s.margin_of_safety)}\n"
        f"  owner_earnings_yield: {pct(s.owner_earnings_yield)}"
    )
    return (
        "【Pabrai 量化 checklist — Dhandho：heads I win, tails I don't lose much】\n"
        "重 MoS 與 OE yield，business simplicity 由 LLM persona 自行判斷（無法量化）。\n"
        "n/a 表示資料源未提供，請在敘事中標記為「未驗證」，不可主觀補值。\n\n"
        + "\n\n".join(sections)
        + "\n\n"
        + summary
    )
