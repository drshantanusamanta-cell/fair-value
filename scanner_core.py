"""Nifty 500 valuation Z-score scanner — data loading (with a disk cache) and per-stock scoring.

No Streamlit here, so it can be tested or run from a script. Data:
  prices        yfinance, batch download, cached per stock (refreshed when older than `price_max_age_h`)
  fundamentals  Screener.in company page, one request per stock with a pause between requests, cached per stock
                (refreshed when older than `fund_max_age_days`; results come out quarterly, so a week is plenty)
"""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import data_sources as ds
import groups as gp
import zcore as zc

CACHE = Path(__file__).parent / ".scan_cache"
CR = 1e7


def _safe(sym: str) -> str:
    return sym.replace("&", "_and_").replace("-", "_").replace("/", "_")


def universe() -> pd.DataFrame:
    df = pd.read_csv(gp.CSV_PATH, dtype=str)
    df["code"] = df["Valuation group"].map(gp.LABEL_TO_CODE)
    return df.rename(columns={"NSE symbol": "symbol", "Company": "company", "NSE industry": "nse_industry"})[
        ["symbol", "company", "nse_industry", "code"]].reset_index(drop=True)


def _fresh(path: Path, max_age: timedelta) -> bool:
    return path.exists() and datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) < max_age


def cache_status() -> dict:
    out = {}
    for kind in ("prices", "screener"):
        files = list((CACHE / kind).glob("*.pkl")) if (CACHE / kind).exists() else []
        out[kind] = {"count": len(files),
                     "newest": max((datetime.fromtimestamp(f.stat().st_mtime) for f in files), default=None),
                     "oldest": min((datetime.fromtimestamp(f.stat().st_mtime) for f in files), default=None)}
    return out


# ───────────────────────────── prices ─────────────────────────────
def load_prices(symbols: list[str], price_max_age_h: float = 12, refresh: bool = True, chunk: int = 40,
                progress: Callable[[float, str], None] | None = None) -> dict[str, pd.Series]:
    folder = CACHE / "prices"
    folder.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for s in symbols:
        f = folder / f"{_safe(s)}.pkl"
        if f.exists() and (not refresh or _fresh(f, timedelta(hours=price_max_age_h))):
            out[s] = pd.read_pickle(f)
        else:
            todo.append(s)
    if todo:
        import yfinance as yf
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            if progress:
                progress(i / len(todo), f"Prices: {i}/{len(todo)} from yfinance")
            try:
                data = yf.download([f"{s}.NS" for s in part], period="15y", interval="1d", group_by="ticker",
                                   auto_adjust=False, progress=False, threads=True)
            except Exception:
                data = None
            for s in part:
                close = None
                try:
                    if data is not None and (f"{s}.NS", "Close") in data.columns:
                        close = data[(f"{s}.NS", "Close")].dropna()
                    elif data is not None and "Close" in data.columns and len(part) == 1:
                        close = data["Close"].dropna()
                except Exception:
                    close = None
                if close is not None and len(close) > 250:
                    close.index = pd.to_datetime(close.index).tz_localize(None) if close.index.tz else pd.to_datetime(close.index)
                    close.to_pickle(folder / f"{_safe(s)}.pkl")
                    out[s] = close
                elif (folder / f"{_safe(s)}.pkl").exists():      # keep yesterday's data rather than nothing
                    out[s] = pd.read_pickle(folder / f"{_safe(s)}.pkl")
            time.sleep(1.0)
    if progress:
        progress(1.0, f"Prices: {len(out)}/{len(symbols)} available")
    return out


# ───────────────────────────── fundamentals ─────────────────────────────
def _fy_years(r: dict) -> int:
    e = ds._annual(ds.screener_row(r.get("pl"), "EPS in Rs", "EPS"))
    return 0 if e is None else int(e.notna().sum())


def _fetch_one_screener(sym: str, pause: float) -> dict:
    r = ds.fetch_screener(sym, True, peers=False)
    if not r.get("ok") and r.get("status") == 429:      # rate-limited: wait once and retry
        time.sleep(30)
        r = ds.fetch_screener(sym, True, peers=False)
    time.sleep(pause)
    # Some companies (e.g. HDFC AMC) have only a few years on Screener's consolidated page but 10+ standalone.
    # The Z needs long history, so if consolidated has < 6 fiscal years of EPS, keep the longer of the two.
    if r.get("ok") and r.get("basis") == "Consolidated" and _fy_years(r) < 6:
        alt = ds.fetch_screener(sym, False, peers=False)
        time.sleep(pause)
        if alt.get("ok") and _fy_years(alt) > _fy_years(r):
            r = alt
    keep = {k: r.get(k) for k in ("ok", "error", "basis", "pl", "bs", "cf", "ratios")}
    keep["fetched"] = datetime.now()
    return keep


def load_screener(symbols: list[str], fund_max_age_days: float = 7, refresh: bool = True, workers: int = 3,
                  pause: float = 1.0, progress: Callable[[float, str], None] | None = None) -> dict[str, dict]:
    folder = CACHE / "screener"
    folder.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for s in symbols:
        f = folder / f"{_safe(s)}.pkl"
        if f.exists() and (not refresh or _fresh(f, timedelta(days=fund_max_age_days))):
            with open(f, "rb") as fh:
                out[s] = pickle.load(fh)
        else:
            todo.append(s)
    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_fetch_one_screener, s, pause): s for s in todo}
            for fut in as_completed(futs):
                s = futs[fut]
                done += 1
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                f = folder / f"{_safe(s)}.pkl"
                if r.get("ok"):
                    with open(f, "wb") as fh:
                        pickle.dump(r, fh)
                    out[s] = r
                elif f.exists():                           # keep the older copy if the refresh failed
                    with open(f, "rb") as fh:
                        out[s] = pickle.load(fh)
                else:
                    out[s] = r
                if progress and (done % 5 == 0 or done == len(todo)):
                    progress(done / len(todo), f"Screener: {done}/{len(todo)} stocks")
    return out


# ───────────────────────────── scoring ─────────────────────────────
def extras_from_screener(scr: dict) -> dict:
    pl, bs, cf = scr.get("pl"), scr.get("bs"), scr.get("cf")
    ext = ds.annual_histories(pl, bs, cf, None, {})
    ext["eps_annual"] = ds._annual(ds.screener_row(pl, "EPS in Rs", "EPS"))
    return ext


def score_one(sym: str, code: str, close: pd.Series | None, scr: dict | None, p: dict) -> dict:
    """One scanner row. p = settings: z_years, band_years, min_years, mid_n, ke, g_term, wacc, tax, mos."""
    lab = gp.GROUPS[code][0] if code in gp.GROUPS else "—"
    row = {"Symbol": sym, "Group": lab, "code": code, "Signal": "Data missing", "rank": 99, "Mcap (₹ Cr)": np.nan,
           "Z weekly": np.nan}
    if close is None or close.dropna().empty:
        row["Error"] = "no price history"
        return row
    if not scr or not scr.get("ok"):
        row["Error"] = f"no Screener data ({(scr or {}).get('error') or 'not fetched'})"
        return row
    try:
        ext = extras_from_screener(scr)
        rat = scr.get("ratios") or {}
        price = float(close.dropna().iloc[-1])
        mcap_cr = rat.get("Market Cap")
        shares = (mcap_cr * CR / rat["Current Price"]) if ds.finite(mcap_cr) and ds.finite(rat.get("Current Price")) \
            and rat["Current Price"] else (float(ext["shares_annual"].iloc[-1]) if ext.get("shares_annual") is not None
                                           and len(ext["shares_annual"]) else float("nan"))
        nd = ext.get("net_debt_annual")
        nd_now = float(nd.iloc[-1]) if nd is not None and len(nd) else 0.0
        methods = [m for m in gp.METHODS.get(code, []) if m in zc.HISTORY_OK]
        hc = zc.composite_history(close, ext, methods, band_years=p["band_years"], min_years=p["min_years"],
                                  mid_n=p["mid_n"], use_mid=code in gp.MID_CYCLE, allow_ev=code not in gp.FIN_GROUPS,
                                  shares_now=shares, net_debt_now=nd_now, minority=0.0, ke=p["ke"], g_term=p["g_term"],
                                  tax=p["tax"], epv_rate=p["wacc"])
        snap = zc.z_snapshot(close, hc["composite"] if not hc.empty else None, p["z_years"])
        sig, rank = zc.signal(snap["z_w"], snap["z_w_min4"], snap["above_sma20"])
        fair = snap["fair"]
        n_used = int(hc["n_methods"].iloc[-1]) if not hc.empty else 0
        row.update({
            "Mcap (₹ Cr)": mcap_cr if ds.finite(mcap_cr) else np.nan, "Price": price,
            "Z weekly": snap["z_w"], "Z daily": snap["z_d"],
            "Z vs 4-wk low": snap["z_w"] - snap["z_w_min4"] if ds.finite(snap["z_w"]) and ds.finite(snap["z_w_min4"]) else np.nan,
            "Above 20-day avg": snap["above_sma20"], "Fair value (history)": fair,
            "Upside %": (fair / price - 1) * 100 if ds.finite(fair) and price else np.nan,
            "Buy line": fair * (1 - p["mos"]) if ds.finite(fair) else np.nan,
            "Methods": f"{n_used}/{len(methods)}", "n_methods": n_used,
            "Z since": snap["z_since"].strftime("%b %Y") if snap["z_since"] is not None else "—",
            "Signal": sig, "rank": rank, "Basis": scr.get("basis") or "", "Error": "",
        })
    except Exception as e:      # one bad stock must not stop the scan
        row["Error"] = f"{type(e).__name__}: {e}"
    return row


def add_size(df: pd.DataFrame) -> pd.DataFrame:
    """Large = top 100 by market cap within the list, Mid = next 150, Small = the rest (SEBI-style ranks)."""
    df = df.copy()
    r = df["Mcap (₹ Cr)"].rank(ascending=False, method="first")
    df["Size"] = np.where(r <= 100, "Large", np.where(r <= 250, "Mid", np.where(r.notna(), "Small", "—")))
    return df


def run_scan(prices: dict[str, pd.Series], funds: dict[str, dict], p: dict, uni: pd.DataFrame | None = None,
             progress: Callable[[float, str], None] | None = None) -> pd.DataFrame:
    uni = universe() if uni is None else uni
    rows = []
    for i, (sym, comp, code) in enumerate(zip(uni["symbol"], uni["company"], uni["code"])):
        row = score_one(sym, code, prices.get(sym), funds.get(sym), p)
        row["Company"] = comp
        rows.append(row)
        if progress and (i % 10 == 0 or i == len(uni) - 1):
            progress((i + 1) / len(uni), f"Scoring {i + 1}/{len(uni)}")
    df = add_size(pd.DataFrame(rows))
    return df.sort_values(["rank", "Z weekly"], na_position="last").reset_index(drop=True)
