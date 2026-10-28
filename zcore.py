"""History-based composite fair value and its Z-score — one engine shared by the Workbench (Summary tab)
and the Nifty 500 Z scanner, so a stock flagged by the scanner shows the same reading when opened.

Only methods that can be rebuilt year by year are used; each uses data public at that date (60-day lag after the
fiscal year-end) and band means use only multiples up to that date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import valuation as v

HISTORY_OK = ["Graham Number", "P/E band (own history)", "P/B band (own history)", "EV/EBITDA band (own history)",
              "EV/Sales band (own history)", "Justified P/B", "EPV"]
EV_METHODS = {"EV/EBITDA band (own history)", "EV/Sales band (own history)", "EPV"}


def _finite(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)


def composite_history(close: pd.Series, ext: dict, methods, *, band_years: float = 5.0, min_years: float = 2.0,
                      mid_n: int = 5, use_mid: bool = False, allow_ev: bool = True, shares_now: float = float("nan"),
                      net_debt_now: float = 0.0, minority: float = 0.0, ke: float = 0.12, g_term: float = 0.05,
                      tax: float = 0.2517, epv_rate: float = 0.11) -> pd.DataFrame:
    """Daily fair value per method + 'composite' (median) + 'n_methods'. Empty DataFrame if nothing can be built.

    ext needs: eps_annual, bvps_annual, shares_annual, net_debt_annual, ebitda_annual, sales_annual_abs, ebit_annual
    (absolute ₹, indexed by fiscal year-end) — as returned by data_sources.annual_histories (+ eps_annual)."""
    if close is None or close.dropna().empty:
        return pd.DataFrame()
    methods = [m for m in methods if m in HISTORY_OK and (allow_ev or m not in EV_METHODS)]
    px = v._clean_prices(close)
    idx = px.index
    eps, bv = ext.get("eps_annual"), ext.get("bvps_annual")
    eps_on, bv_on = v.on_dates(eps, idx), v.on_dates(bv, idx)
    nd_now = net_debt_now if _finite(net_debt_now) else 0.0
    fair: dict[str, pd.Series] = {}

    def band_fair(b, kind):
        s = b["series"]
        mean_mult = v.rolling_band_fair(pd.Series(1.0, index=s.index), s["multiple"], band_years, min_years)
        if kind == "ps":
            return s["metric"] * mean_mult
        return (s["metric"] * mean_mult - s["net_debt"] - minority) / s["shares"]

    if "Graham Number" in methods and eps is not None and not eps.dropna().empty:
        e3_on = v.on_dates(eps.dropna().rolling(3, min_periods=3).mean().dropna(), idx)
        fair["Graham Number"] = np.sqrt((22.5 * e3_on * bv_on).where((e3_on > 0) & (bv_on > 0)))
    if "P/E band (own history)" in methods and eps is not None:
        b = v.per_share_band(px, eps, band_years, cap=150.0)
        if b:
            fair["P/E band (own history)"] = band_fair(b, "ps")
    if "P/B band (own history)" in methods and bv is not None:
        b = v.per_share_band(px, bv, band_years, cap=30.0)
        if b:
            fair["P/B band (own history)"] = band_fair(b, "ps")
    if _finite(shares_now) and shares_now > 0:
        if "EV/EBITDA band (own history)" in methods:
            eb = v.rolling_fy_mean(ext.get("ebitda_annual"), mid_n) if use_mid else ext.get("ebitda_annual")
            if eb is not None and not eb.dropna().empty:
                b = v.ev_band(px, eb, ext.get("shares_annual"), shares_now, ext.get("net_debt_annual"), nd_now,
                              minority, band_years, cap=60.0)
                if b:
                    fair["EV/EBITDA band (own history)"] = band_fair(b, "ev")
        if "EV/Sales band (own history)" in methods and ext.get("sales_annual_abs") is not None:
            b = v.ev_band(px, ext["sales_annual_abs"], ext.get("shares_annual"), shares_now, ext.get("net_debt_annual"),
                          nd_now, minority, band_years, cap=50.0)
            if b:
                fair["EV/Sales band (own history)"] = band_fair(b, "ev")
        if "EPV" in methods and ext.get("ebit_annual") is not None and epv_rate > 0:
            ebit_on = v.on_dates(ext["ebit_annual"], idx)
            nd_on = v.on_dates(ext.get("net_debt_annual"), idx).bfill().fillna(nd_now)
            sh_on = v.on_dates(ext.get("shares_annual"), idx).fillna(shares_now)
            fair["EPV"] = ((ebit_on * (1 - tax) / epv_rate - nd_on - minority) / sh_on).where(ebit_on > 0)
    if "Justified P/B" in methods and ke > g_term and bv is not None:
        roe_on = eps_on / bv_on                     # EPS ÷ year-end book: approximate ROE
        fair["Justified P/B"] = (bv_on * (roe_on - g_term) / (ke - g_term)).where((roe_on > g_term) & (bv_on > 0))
    return v.history_composite(fair)


def z_snapshot(close: pd.Series, composite: pd.Series, z_years: float = 3.0) -> dict:
    """Weekly and daily Z plus the simple timing checks the scanner shows."""
    out = {"z_w": np.nan, "z_w_prev": np.nan, "z_w_min4": np.nan, "z_d": np.nan, "z_since": None,
           "above_sma20": None, "fair": np.nan}
    if composite is None or composite.dropna().empty:
        return out
    w = v.gap_zscore(close, composite, z_years, "W")
    d = v.gap_zscore(close, composite, z_years, "D")
    zw, zd = w["z"].dropna(), d["z"].dropna()
    if not zw.empty:
        out["z_w"] = float(zw.iloc[-1])
        out["z_w_prev"] = float(zw.iloc[-3]) if len(zw) >= 3 else np.nan       # two weeks earlier
        out["z_w_min4"] = float(zw.iloc[-5:-1].min()) if len(zw) >= 5 else np.nan   # low of the 4 weeks before
        out["z_since"] = zw.index[0]
    if not zd.empty:
        out["z_d"] = float(zd.iloc[-1])
    px = v._clean_prices(close)
    if len(px) >= 20:
        out["above_sma20"] = bool(px.iloc[-1] > px.iloc[-20:].mean())
    out["fair"] = float(composite.dropna().iloc[-1])
    return out


def signal(z_w: float, z_w_min4: float, above_sma20, strong: float = -1.5, value: float = -1.0,
           turn: float = 0.25) -> tuple[str, int]:
    """Label + sort rank. 'Turning up' = weekly Z at least `turn` above its low of the previous 4 weeks AND
    the daily close above its 20-day average — a simple first check, not a full entry trigger."""
    if not _finite(z_w):
        return "No Z yet", 9
    turning = _finite(z_w_min4) and z_w - z_w_min4 >= turn and bool(above_sma20)
    if z_w <= strong:
        return ("Strong value · turning up", 1) if turning else ("Strong value", 2)
    if z_w <= value:
        return ("Value · turning up", 3) if turning else ("Value", 4)
    if z_w >= 2:
        return "Very rich", 8
    if z_w >= 1:
        return "Rich", 7
    return "Neutral", 6
