"""Pure valuation maths for the Fair Value Workbench.

All monetary inputs are absolute rupees (not crores); per-share outputs are rupees per share.
Rates and growths are decimals (0.12 = 12%). Nothing here touches the network or Streamlit,
so every function can be unit-tested in isolation.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.optimize import brentq
except Exception:  # pragma: no cover - scipy is in requirements, bisection fallback only
    brentq = None


def _ok(*vals) -> bool:
    return all(v is not None and isinstance(v, (int, float, np.floating)) and math.isfinite(v) for v in vals)


# ───────────────────────────── Cost of capital ─────────────────────────────
def capm(rf: float, beta: float, erp: float) -> float:
    return rf + beta * erp


def wacc(ke: float, kd_pre_tax: float, tax: float, equity_value: float, debt: float) -> float:
    debt = max(debt or 0.0, 0.0)
    v = equity_value + debt
    if not _ok(ke, kd_pre_tax, tax, equity_value) or v <= 0:
        return ke
    return equity_value / v * ke + debt / v * kd_pre_tax * (1 - tax)


# ───────────────────────────── Growth paths ─────────────────────────────
def growth_path(g1: float, g_term: float, years: int, fade: bool = True) -> list[float]:
    """Year-by-year growth. With fade, year 1 = g1 and the final explicit year = g_term (linear)."""
    if years <= 1 or not fade:
        return [g1] * years
    return [g1 - (g1 - g_term) * t / (years - 1) for t in range(years)]


def cagr(first: float, last: float, years: float) -> float | None:
    if not _ok(first, last, years) or first <= 0 or last <= 0 or years <= 0:
        return None
    return (last / first) ** (1 / years) - 1


def series_cagr(values: Sequence[float], max_years: int = 5) -> tuple[float | None, int]:
    """CAGR over the last `max_years` intervals of a positive annual series (oldest → newest)."""
    v = [x for x in values if _ok(x)]
    if len(v) < 3:
        return None, 0
    n = min(max_years, len(v) - 1)
    return cagr(v[-1 - n], v[-1], n), n


# ───────────────────────────── DCF ─────────────────────────────
def dcf(base_cf: float, growths: Sequence[float], rate: float, g_term: float,
        net_debt: float = 0.0, shares: float = 1.0, mid_year: bool = False) -> dict:
    """Two-stage DCF. `base_cf` is the last actual (year-0) cash flow; growths apply from year 1."""
    if rate <= g_term:
        raise ValueError("Discount rate must exceed terminal growth.")
    rows, pv, cf = [], 0.0, base_cf
    for t, g in enumerate(growths, start=1):
        cf *= 1 + g
        df = 1 / (1 + rate) ** (t - 0.5 if mid_year else t)
        pv += cf * df
        rows.append({"Year": t, "Growth": g, "Cash flow": cf, "Discount factor": df, "PV": cf * df})
    n = len(growths)
    tv = cf * (1 + g_term) / (rate - g_term)
    pv_tv = tv / (1 + rate) ** (n - 0.5 if mid_year else n)
    ev = pv + pv_tv
    equity = ev - (net_debt or 0.0)
    return {
        "table": pd.DataFrame(rows),
        "pv_explicit": pv,
        "terminal_value": tv,
        "pv_terminal": pv_tv,
        "enterprise_value": ev,
        "equity_value": equity,
        "per_share": equity / shares if shares else float("nan"),
        "tv_share": pv_tv / ev if ev else float("nan"),
    }


def dcf_sensitivity(base_cf: float, g1: float, years: int, fade: bool, rates: Iterable[float],
                    g_terms: Iterable[float], net_debt: float, shares: float, mid_year: bool = False) -> pd.DataFrame:
    rates, g_terms = list(rates), list(g_terms)
    grid = np.full((len(rates), len(g_terms)), np.nan)
    for i, rr in enumerate(rates):
        for j, gt in enumerate(g_terms):
            if rr > gt:
                grid[i, j] = dcf(base_cf, growth_path(g1, gt, years, fade), rr, gt, net_debt, shares, mid_year)["per_share"]
    return pd.DataFrame(grid, index=[f"{x*100:.1f}%" for x in rates], columns=[f"{x*100:.1f}%" for x in g_terms])


def reverse_dcf(price: float, shares: float, net_debt: float, base_cf: float, years: int,
                rate: float, g_term: float, fade: bool = True, mid_year: bool = False,
                lo: float = -0.5, hi: float = 1.5) -> float | None:
    """Constant (or fading) stage-1 growth the market price implies. None if no solution in [lo, hi]."""
    if not _ok(price, shares, base_cf, rate, g_term) or base_cf <= 0 or shares <= 0 or rate <= g_term:
        return None

    def f(g):
        return dcf(base_cf, growth_path(g, g_term, years, fade), rate, g_term, net_debt, shares, mid_year)["per_share"] - price

    a, b = f(lo), f(hi)
    if a * b > 0:
        return None
    if brentq is not None:
        return float(brentq(f, lo, hi, xtol=1e-7, maxiter=200))
    for _ in range(100):  # bisection fallback
        m = (lo + hi) / 2
        if f(m) > 0:
            hi = m
        else:
            lo = m
    return (lo + hi) / 2


# ───────────────────────────── Dividend & residual income ─────────────────────────────
def gordon(dps: float, ke: float, g: float) -> float | None:
    if not _ok(dps, ke, g) or dps <= 0 or ke <= g:
        return None
    return dps * (1 + g) / (ke - g)


def ddm_two_stage(dps: float, g1: float, ke: float, g_term: float, years: int, fade: bool = True) -> float | None:
    if not _ok(dps) or dps <= 0 or ke <= g_term:
        return None
    return dcf(dps, growth_path(g1, g_term, years, fade), ke, g_term)["per_share"]


def residual_income(bvps: float, roe: float, ke: float, payout: float, years: int = 10,
                    fade: bool = True, terminal_spread: float = 0.0, g_term: float = 0.05) -> dict | None:
    """Book value + PV of (ROE − Ke) × opening book. ROE fades linearly to Ke + terminal_spread.

    If terminal_spread > 0 the final residual income persists as a growing perpetuity at g_term.
    """
    if not _ok(bvps, roe, ke, payout) or bvps <= 0:
        return None
    target = ke + terminal_spread
    b, value, rows = bvps, bvps, []
    ri = 0.0
    for t in range(1, years + 1):
        roe_t = roe - (roe - target) * (t - 1) / (years - 1) if fade and years > 1 else roe
        eps = roe_t * b
        ri = (roe_t - ke) * b
        pv = ri / (1 + ke) ** t
        value += pv
        rows.append({"Year": t, "ROE": roe_t, "Opening BVPS": b, "EPS": eps, "Residual income": ri, "PV": pv})
        b = b + eps * (1 - min(max(payout, 0.0), 1.0))
    tv_pv = 0.0
    if terminal_spread > 0 and ke > g_term:
        tv_pv = ri * (1 + g_term) / (ke - g_term) / (1 + ke) ** years
        value += tv_pv
    return {"per_share": value, "table": pd.DataFrame(rows), "pv_terminal": tv_pv}


def justified_pb(roe: float, ke: float, g: float) -> float | None:
    if not _ok(roe, ke, g) or ke <= g:
        return None
    return (roe - g) / (ke - g)


# ───────────────────────────── Formula methods ─────────────────────────────
def graham_number(eps: float, bvps: float) -> float | None:
    if not _ok(eps, bvps) or eps <= 0 or bvps <= 0:
        return None
    return math.sqrt(22.5 * eps * bvps)


def graham_growth(eps: float, g: float, bond_yield_pct: float) -> float | None:
    """Revised Graham formula: EPS × (8.5 + 2g) × 4.4 / Y, with g and Y in percent."""
    if not _ok(eps, g, bond_yield_pct) or eps <= 0 or bond_yield_pct <= 0:
        return None
    return eps * (8.5 + 2 * g * 100) * 4.4 / bond_yield_pct


def epv(ebit: float, tax: float, cost_of_capital: float, net_debt: float, shares: float) -> float | None:
    """Earnings Power Value: no-growth value of today's normalised after-tax operating earnings."""
    if not _ok(ebit, tax, cost_of_capital, shares) or ebit <= 0 or shares <= 0 or cost_of_capital <= 0:
        return None
    return (ebit * (1 - tax) / cost_of_capital - (net_debt or 0.0)) / shares


# ───────────────────────────── Multiples ─────────────────────────────
def pe_band(prices: pd.Series, eps_annual: pd.Series, years: float = 5.0, report_lag_days: int = 60,
            cap: float = 150.0) -> dict | None:
    """Historical P/E band using annual EPS made public `report_lag_days` after fiscal year-end.

    prices: Series indexed by date. eps_annual: Series indexed by fiscal year-end date.
    """
    if prices is None or eps_annual is None or prices.empty or eps_annual.dropna().empty:
        return None
    p = prices.dropna().copy()
    p.index = pd.to_datetime(p.index).tz_localize(None) if getattr(p.index, "tz", None) else pd.to_datetime(p.index)
    e = eps_annual.dropna().copy()
    e.index = pd.to_datetime(e.index) + pd.Timedelta(days=report_lag_days)
    e = e.sort_index()
    eps_on_date = e.reindex(p.index.union(e.index)).ffill().reindex(p.index)
    pe = (p / eps_on_date).where(eps_on_date > 0).clip(upper=cap)
    start = p.index.max() - pd.Timedelta(days=int(365.25 * years))
    window = pe[pe.index >= start].dropna()
    if len(window) < 20:
        return None
    mean, sd = float(window.mean()), float(window.std())
    current = float(window.iloc[-1])
    return {
        "series": pd.DataFrame({"price": p, "eps": eps_on_date, "pe": pe}),
        "mean": mean, "std": sd, "median": float(window.median()),
        "current": current, "z": (current - mean) / sd if sd > 0 else float("nan"),
        "window_start": start,
    }


def median_of(values: Iterable[float]) -> float | None:
    v = [x for x in values if _ok(x) and x > 0]
    return float(np.median(v)) if v else None


# ───────────────────────────── SOTP ─────────────────────────────
def sotp(segments: pd.DataFrame, holdings: pd.DataFrame, holding_discount: float,
         net_debt: float, shares: float) -> dict | None:
    """segments: columns Metric (₹ Cr), Multiple. holdings: column Value (₹ Cr) — attributable stake value."""
    seg_val = 0.0
    if segments is not None and not segments.empty:
        s = segments.dropna(subset=["Metric (₹ Cr)", "Multiple"])
        seg_val = float((s["Metric (₹ Cr)"] * s["Multiple"]).sum()) * 1e7
    hold_val = 0.0
    if holdings is not None and not holdings.empty:
        h = holdings.dropna(subset=["Value (₹ Cr)"])
        hold_val = float(h["Value (₹ Cr)"].sum()) * 1e7 * (1 - holding_discount)
    if seg_val + hold_val <= 0 or not shares:
        return None
    equity = seg_val + hold_val - (net_debt or 0.0)
    return {"segments": seg_val, "holdings": hold_val, "equity": equity, "per_share": equity / shares}


# ───────────────────────────── Auto assumptions ─────────────────────────────
def regression_beta(stock_close: pd.Series, index_close: pd.Series, years: float = 2.0,
                    blume: bool = True) -> tuple[float | None, int]:
    """Beta of weekly returns vs an index over the last `years`. Blume-adjusted (0.67·β + 0.33) by default.
    Returns (beta, number of weekly observations)."""
    try:
        s = stock_close.dropna().copy(); i = index_close.dropna().copy()
        for x in (s, i):
            x.index = pd.to_datetime(x.index)
            if getattr(x.index, "tz", None) is not None:
                x.index = x.index.tz_localize(None)
        df = pd.concat([s.resample("W-FRI").last().pct_change(), i.resample("W-FRI").last().pct_change()], axis=1).dropna()
        df = df[df.index >= df.index.max() - pd.Timedelta(days=int(365.25 * years))]
        if len(df) < 40:
            return None, len(df)
        c = np.cov(df.iloc[:, 0], df.iloc[:, 1])
        b = c[0, 1] / c[1, 1]
        return float(0.67 * b + 0.33 if blume else b), len(df)
    except Exception:
        return None, 0


SIZE_RULES = {  # approximate SEBI/AMFI cut-offs by market cap (₹ crore); rule-of-thumb premia
    "Large cap": {"min_cr": 100_000, "erp": 5.75, "mos": 25},
    "Mid cap":   {"min_cr": 30_000,  "erp": 6.25, "mos": 30},
    "Small cap": {"min_cr": 0,       "erp": 6.75, "mos": 35},
}


def auto_assumptions(rf: float, mcap: float | None, growth: float | None, beta_reg: float | None,
                     roe: float | None) -> dict:
    """Per-stock starting assumptions. rf and growth/roe are decimals; mcap in ₹. Returns values + reasons.

    These are documented rules of thumb, not market-derived 'optimal' values.
    """
    cr = mcap / 1e7 if _ok(mcap) else None
    size = next((k for k, r in SIZE_RULES.items() if cr is not None and cr >= r["min_cr"]), "Small cap" if cr is not None else "Mid cap")
    rule = SIZE_RULES[size]
    size_note = f"{size} (₹{cr:,.0f} Cr)" if cr is not None else "market cap unknown → treated as mid cap"

    if beta_reg is not None and _ok(beta_reg):
        beta, beta_note = float(min(max(beta_reg, 0.6), 1.8)), f"2-yr weekly vs NIFTY 50, Blume-adjusted ({beta_reg:.2f}"
        beta_note += ", clipped)" if beta != beta_reg else ")"
    else:
        beta, beta_note = 1.0, "regression unavailable → market beta 1.0"

    g_term = max(3.0, min(5.0, rf * 100 - 2.0)) / 100
    g = growth if _ok(growth) else None
    years = 10 if g is not None and g >= 0.15 else 7 if g is not None and g >= 0.08 else 5
    g_note = f"historical growth {g * 100:.1f}%" if g is not None else "growth unknown"

    ke = rf + beta * rule["erp"] / 100
    spread = 0.0
    if _ok(roe) and roe > ke:
        spread = min(round((roe - ke) / 2 * 200) / 200, 0.05)   # half today's excess return, to nearest 0.5 pp, max 5 pp
    return {
        "erp": rule["erp"] / 100, "beta": round(beta, 2), "g_term": g_term, "years": years,
        "fade": True, "mid_year": True, "mos": rule["mos"] / 100, "ri_spread": spread, "size": size,
        "notes": {
            "erp": f"{size_note} → ERP {rule['erp']:.2f}%",
            "beta": beta_note,
            "g_term": f"min(5%, G-Sec − 2 pp), floor 3% → {g_term * 100:.2f}%",
            "years": f"{g_note} → {years} yrs (≥15%: 10 · ≥8%: 7 · else 5)",
            "mos": f"{size} → {rule['mos']}%",
            "ri_spread": (f"half of ROE {roe * 100:.1f}% − Ke {ke * 100:.1f}% → {spread * 100:.1f} pp (max 5)"
                          if _ok(roe) else "ROE unknown → 0"),
        },
    }


# ───────────────────────────── Composite & verdict ─────────────────────────────
def composite(results: dict[str, dict], include: Iterable[str]) -> dict | None:
    """results: name → {"base": float, "low": float, "high": float}. Median of included bases."""
    bases = [results[k]["base"] for k in include if k in results and _ok(results[k].get("base")) and results[k]["base"] > 0]
    if not bases:
        return None
    arr = np.array(bases)
    med, mean = float(np.median(arr)), float(arr.mean())
    cv = float(arr.std() / mean) if len(arr) > 1 and mean > 0 else float("nan")
    n = len(arr)
    if n >= 4 and cv < 0.25:
        conf = "HIGH"
    elif n >= 3 and cv < 0.5:
        conf = "MEDIUM"
    else:
        conf = "LOW"
    return {"fair_value": med, "mean": mean, "low": float(np.percentile(arr, 25)), "high": float(np.percentile(arr, 75)),
            "n": n, "cv": cv, "confidence": conf}


def verdict(price: float, fair: float, mos: float) -> tuple[str, str]:
    if not _ok(price, fair) or fair <= 0:
        return "NO DATA", "gray"
    if price <= fair * (1 - mos):
        return f"BUY ZONE — price ≥{mos*100:.0f}% below fair value", "green"
    if price <= fair:
        return "BELOW FAIR VALUE — margin of safety thinner than required", "lightgreen"
    if price <= fair * 1.2:
        return "FAIRLY VALUED / MILD PREMIUM", "orange"
    return "OVERVALUED — >20% above fair value", "red"
