"""Shared helpers used by master scoring modules.

Centralised so the same percent/currency formatting and verdict thresholds
appear identically across the 13 masters' output blocks — preventing the
LLM from inferring spurious signals from formatting differences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from alpha_council.providers.base import FinancialMetrics, LineItem


def pct(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.{digits}f}%"


def num(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.{digits}f}"


def money(v: float | None) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 1e9:
        return f"{v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.2f}K"
    return f"{v:.2f}"


def latest(items: Sequence[FinancialMetrics] | None) -> FinancialMetrics | None:
    if not items:
        return None
    return max(items, key=lambda f: f.period_end)


def line_item_series(items: Sequence[LineItem] | None, name: str) -> list[LineItem]:
    if not items:
        return []
    return sorted(
        (li for li in items if li.name == name),
        key=lambda li: li.period_end,
        reverse=True,
    )


def compute_owner_earnings(
    latest_fm: FinancialMetrics | None,
    line_items: Sequence[LineItem] | None,
) -> tuple[float | None, str]:
    """Buffett's owner earnings ≈ OCF − maintenance CapEx.

    Returns `(value, derivation_string)`. The derivation string is what the
    LLM-side scorecard surfaces so it can reason about what was used. Order
    of preference:
      1. OCF (latest period) − abs(CapEx) — most faithful to Buffett's defn.
      2. Latest FCF from FinancialMetrics — yfinance's already-derived value.
      3. None — caller must mark criterion unverified.

    `maintenance` vs `growth` CapEx isn't split anywhere in our data, so we
    use full CapEx — conservative bias is acceptable for screening.
    """
    items_list = list(line_items or [])
    ocf_series = line_item_series(items_list, "operating_cash_flow")
    capex_series = line_item_series(items_list, "capital_expenditure")
    ocf = ocf_series[0].value if ocf_series and ocf_series[0].value is not None else None
    capex = capex_series[0].value if capex_series and capex_series[0].value is not None else None
    if ocf is not None:
        oe = ocf - (abs(capex) if capex is not None else 0)
        return oe, f"OE = OCF({money(ocf)}) − CapEx({money(capex)})"
    if latest_fm is not None and latest_fm.free_cash_flow is not None:
        return latest_fm.free_cash_flow, f"OE ≈ FCF = {money(latest_fm.free_cash_flow)}"
    return None, "OCF / FCF both unavailable"


def verdict_from_score(total: float, max_total: float) -> str:
    """Map normalized score to a single-word stance the LLM can anchor on."""
    if max_total <= 0:
        return "n/a"
    ratio = total / max_total
    if ratio >= 0.75:
        return "strong-fit"
    if ratio >= 0.50:
        return "qualified-fit"
    if ratio >= 0.30:
        return "borderline"
    return "fails-screen"


@dataclass(frozen=True)
class ScoreLine:
    """One row in a master's scorecard table."""

    label: str
    score: float
    max_score: float
    detail: str


def format_scorecard(title: str, lines: list[ScoreLine]) -> str:
    """Render the scorecard as a fenced markdown block.

    Layout is intentionally compact so it embeds well in system_instruction
    without dominating prompt budget — masters typically need ~30 lines.
    """
    rows = []
    for l in lines:
        rows.append(f"  - {l.label}: {l.score:.1f} / {l.max_score:.1f}  — {l.detail}")
    total = sum(l.score for l in lines)
    cap = sum(l.max_score for l in lines)
    return (
        f"### {title}\n"
        f"  total: {total:.1f} / {cap:.1f}  ({verdict_from_score(total, cap)})\n"
        + "\n".join(rows)
    )
