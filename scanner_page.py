"""Nifty 500 Z Scanner — opened from the page switch at the top of the Workbench sidebar (app.py runs this file).
Ranks the Nifty 500 by how cheap price is versus its history-based fair value
(same engine as the Workbench's Summary tab and the same valuation groups as the Pine indicator)."""
from __future__ import annotations

import pickle
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import groups as gp  # noqa: E402
import scanner_core as sc  # noqa: E402

if __name__ == "__main__":   # run on its own (streamlit run scanner_page.py); inside app.py the page config is already set
    st.set_page_config(page_title="Nifty 500 Z Scanner", page_icon="🔎", layout="wide")
LAST = sc.CACHE / "last_scan.pkl"

st.title("🔎 Nifty 500 Z Scanner")
st.caption("Finds stocks trading unusually cheap against their own history-based fair value — the same weekly Z the "
           "Workbench shows. A value signal is a starting point for a swing setup, not a buy signal: confirm on the "
           "daily chart and in the Workbench before entering.")

# ───────────────────────────── settings ─────────────────────────────
with st.sidebar:
    st.header("Scanner settings")
    z_years = st.slider("Z lookback (years)", 1.0, 6.0, 3.0, 0.5, help="Same default as the Pine script and Workbench.")
    band_years = st.slider("Band window (years)", 2, 10, 5, help="Averaging window for the P/E, P/B, EV bands.")
    min_years = st.slider("Band history needed before a band counts (years)", 1.0, 5.0, 2.0, 0.5)
    mid_n = st.slider("Mid-cycle EBITDA years (Cyclical group)", 3, 10, 5)
    ke = st.number_input("Cost of equity % (justified P/B)", 6.0, 25.0, 12.0, 0.25) / 100
    wacc = st.number_input("WACC % (EPV)", 6.0, 25.0, 11.0, 0.25) / 100
    g_term = st.number_input("Terminal / book growth %", 0.0, 10.0, 5.0, 0.25) / 100
    mos = st.slider("Margin of safety for the buy line %", 0, 50, 25) / 100
    st.caption("Defaults match the Pine indicator. These are single values for all 500 stocks — open a stock in the "
               "Workbench for stock-specific assumptions.")
P = {"z_years": z_years, "band_years": band_years, "min_years": min_years, "mid_n": mid_n, "ke": ke,
     "g_term": g_term, "wacc": wacc, "tax": 0.2517, "mos": mos}

# ───────────────────────────── data & run ─────────────────────────────
uni = sc.universe()
status = sc.cache_status()
c1, c2, c3 = st.columns([1.2, 1.2, 2])
c1.metric("Prices cached", f"{status['prices']['count']} / {len(uni)}",
          f"newest {status['prices']['newest']:%d %b %H:%M}" if status["prices"]["newest"] else None, delta_color="off")
c2.metric("Fundamentals cached", f"{status['screener']['count']} / {len(uni)}",
          f"oldest {status['screener']['oldest']:%d %b}" if status["screener"]["oldest"] else None, delta_color="off")
with c3:
    full = st.button("⬇️ Update data & scan", type="primary", width="stretch",
                     help="Downloads prices older than 12 h and fundamentals older than 7 days, then scores all 500. "
                          "First run ≈ 10–15 min (Screener is fetched politely, with a pause after each stock); "
                          "later runs take a few minutes.")
    quick = st.button("⚡ Re-score from cached data", width="stretch",
                      help="No downloads — use after changing the settings on the left.")


def _run(refresh: bool):
    bar = st.progress(0.0, "Starting…")
    syms = list(uni["symbol"])
    prices = sc.load_prices(syms, refresh=refresh, progress=lambda f, m: bar.progress(min(f, 1.0) * 0.3, m))
    funds = sc.load_screener(syms, refresh=refresh, progress=lambda f, m: bar.progress(0.3 + min(f, 1.0) * 0.5, m))
    df = sc.run_scan(prices, funds, P, uni, progress=lambda f, m: bar.progress(0.8 + min(f, 1.0) * 0.2, m))
    bar.empty()
    res = {"df": df, "when": datetime.now(), "settings": P}
    sc.CACHE.mkdir(parents=True, exist_ok=True)
    with open(LAST, "wb") as fh:
        pickle.dump(res, fh)
    st.session_state["scan"] = res


if full:
    _run(refresh=True)
elif quick:
    if status["prices"]["count"] == 0 or status["screener"]["count"] == 0:
        st.warning("No cached data yet — run **Update data & scan** first.")
    else:
        _run(refresh=False)

if "scan" not in st.session_state and LAST.exists():
    try:
        with open(LAST, "rb") as fh:
            st.session_state["scan"] = pickle.load(fh)
    except Exception:
        pass

scan = st.session_state.get("scan")
if not scan:
    st.info("No scan yet. Click **Update data & scan** — the first run downloads data for all 500 stocks.")
    st.stop()

df = scan["df"].copy()
if scan["settings"] != P:
    st.warning("Settings changed since this scan — click **Re-score from cached data** to apply them.")
st.caption(f"Last scan {scan['when']:%d %b %Y %H:%M} · Nifty 500 list of {gp.LIST_DATE}")

# ───────────────────────────── filters ─────────────────────────────
ok = df[df["Error"].fillna("") == ""]
f1, f2, f3, f4 = st.columns([1, 1.6, 1.6, 1])
sizes = f1.multiselect("Size", ["Large", "Mid", "Small"], default=["Large", "Mid"])
grps = f2.multiselect("Valuation group", [g[0] for g in gp.GROUPS.values()], default=[g[0] for g in gp.GROUPS.values()])
SIGS = ["Strong value · turning up", "Strong value", "Value · turning up", "Value", "Neutral", "Rich", "Very rich", "No Z yet"]
sigs = f3.multiselect("Signal", SIGS, default=SIGS[:4])
min_m = f4.number_input("Min. methods", 1, 6, 2, help="Methods with enough history behind the fair value.")
view = ok[ok["Size"].isin(sizes) & ok["Group"].isin(grps) & ok["Signal"].isin(sigs) & (ok["n_methods"] >= min_m)]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Scored", f"{len(ok)} / {len(df)}")
m2.metric("Value zone (weekly Z ≤ −1)", int((ok["Z weekly"] <= -1).sum()))
m3.metric("…and turning up", int(ok["Signal"].str.contains("turning up").sum()))
m4.metric("Shown with your filters", len(view))

# ───────────────────────────── table ─────────────────────────────
view = view.assign(Workbench=[f"/?s={quote(s)}" for s in view["Symbol"]])
cols = ["Signal", "Symbol", "Company", "Size", "Group", "Z weekly", "Z daily", "Z vs 4-wk low", "Above 20-day avg",
        "Price", "Fair value (history)", "Upside %", "Buy line", "Methods", "Z since", "Workbench"]
st.dataframe(
    view[cols], hide_index=True, width="stretch", height=min(640, 38 * (len(view) + 1)),
    column_config={
        "Z weekly": st.column_config.NumberColumn(format="%+.2f"),
        "Z daily": st.column_config.NumberColumn(format="%+.2f"),
        "Z vs 4-wk low": st.column_config.NumberColumn("Z vs 4-wk low", format="%+.2f",
                                                       help="How far weekly Z has risen from its low of the previous 4 weeks"),
        "Above 20-day avg": st.column_config.CheckboxColumn("Close > 20-day avg"),
        "Price": st.column_config.NumberColumn(format="₹%.1f"),
        "Fair value (history)": st.column_config.NumberColumn(format="₹%.1f"),
        "Upside %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Buy line": st.column_config.NumberColumn(format="₹%.1f"),
        "Workbench": st.column_config.LinkColumn("Open", display_text="Workbench ↗"),
    })

c_dl, _ = st.columns([1, 3])
c_dl.download_button("⬇️ Download CSV", view[cols[:-1]].to_csv(index=False).encode("utf-8"),
                     file_name=f"nifty500_z_scan_{scan['when']:%Y%m%d}.csv", mime="text/csv")

# ───────────────────────────── chart ─────────────────────────────
plot = ok[ok["Size"].isin(sizes) & ok["Group"].isin(grps) & ok["Z weekly"].notna() & ok["Upside %"].notna()]
if not plot.empty:
    fig = go.Figure()
    for code, (lab, col) in gp.GROUPS.items():
        d = plot[plot["code"] == code]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(x=d["Z weekly"], y=d["Upside %"].clip(-80, 200), mode="markers", name=lab,
                                 marker=dict(color=col, size=8, opacity=0.8), text=d["Symbol"],
                                 hovertemplate="%{text}<br>Z %{x:+.2f}<br>Upside %{y:+.0f}%<extra></extra>"))
    for x, dash in [(-1, "dot"), (-1.5, "dash")]:
        fig.add_vline(x=x, line=dict(color="#2E7D32", width=1, dash=dash))
    fig.update_layout(title="All scored stocks (size and group filters applied) — left = cheaper than usual",
                      xaxis_title="Weekly Z (price vs history-based fair value)", yaxis_title="Upside to fair value %",
                      height=460, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

with st.expander("How to read the signals"):
    st.markdown(
        "- **Weekly Z** = (ln(price ÷ fair value) − its 3-year mean) ÷ its 3-year standard deviation. "
        "≤ −1 = **Value**, ≤ −1.5 = **Strong value**, by this stock's own history.\n"
        "- **Turning up** = weekly Z has risen at least 0.25 from its low of the previous 4 weeks **and** the daily "
        "close is above its 20-day average — a simple first check that the fall has paused. Still confirm the entry "
        "on the daily chart.\n"
        "- **Fair value (history)** is rebuilt only from methods with yearly history for the stock's group "
        "(Graham Number, P/E, P/B, EV/EBITDA, EV/Sales bands, justified P/B, EPV). DCF, DDM, residual income and peer "
        "values are not in the scan, so the Workbench headline composite can differ.\n"
        "- **Z and Upside can disagree.** Upside compares price with fair value directly; Z compares today's gap "
        "with the stock's *usual* gap. A stock that always trades 40% above its fair value but is now only 10% above "
        "shows negative upside and a strongly negative (cheap) Z. For swing entries Z is the signal; upside tells you "
        "how far the stock sits from intrinsic value.\n"
        "- **Size**: Large = top 100 by market cap within the Nifty 500, Mid = next 150, Small = the rest.\n"
        "- **Methods** shows how many methods had enough history; one method alone is a weak basis.")
with st.expander("Data notes and limits"):
    st.markdown(
        "- Prices: yfinance; fundamentals: Screener.in consolidated pages (standalone if no consolidated). Cached in "
        "`.scan_cache/`; prices refresh after 12 h, fundamentals after 7 days.\n"
        "- In the scan, net debt = Screener borrowings (cash is not on Screener's balance sheet) and minority interest "
        "is not deducted. Both are applied consistently to past and present EVs, so the bias largely cancels in the "
        "Z, but the fair-value level can differ from the Workbench.\n"
        "- Cost of equity, WACC and growth are one setting for all stocks.\n"
        "- The Nifty 500 list changes at each NSE rebalance (March, September); stocks that left the index are not "
        "scanned until the list is refreshed.\n"
        "- Educational tool, not investment advice.")
err = df[df["Error"].fillna("") != ""]
if not err.empty:
    with st.expander(f"⚠️ {len(err)} stocks could not be scored"):
        st.dataframe(err[["Symbol", "Company", "Group", "Error"]], hide_index=True, width="stretch")
