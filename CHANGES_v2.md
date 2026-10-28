# Fair Value Workbench v2 — 28 Oct 2026

Aligned with the Pine indicator *Shantanu's Valuation Matrix with Z-Score* (SVM-Z v1.1).

## New files
- `groups.py` — the 8 valuation groups, their default method sets, and the lookup order
  (your sidebar choice → Nifty 500 list → yfinance industry → yfinance sector → General).
- `nifty500_valuation_groups.csv` — hand-checked group for all 500 Nifty 500 stocks (NSE list of 28 Oct 2026).
  Refresh after each NSE rebalance (March, September).
- `tests/test_valuation.py` — unit tests (`python -m pytest tests`).
- `tests/run_app_mocked.py` — headless run of the whole app on saved Screener pages.

## What changed
| Area | Before | Now |
|---|---|---|
| Financials check | Keyword search ("finance", "capital markets"…) — AMCs, exchanges, brokers lost DCF/EPV | 8 valuation groups; DCF/EPV/EV methods off only for Lender, Insurer, Holding co |
| Default composite | All methods (minus analyst target; minus DCF/EPV/DDM for "financials") | The group's method set; Residual income for lenders/insurers; SOTP for holding cos |
| Own-history bands | P/E only | + P/B, EV/EBITDA (mid-cycle EBITDA for Cyclical), EV/Sales |
| Swing signal | P/E Z only | History-based composite + 3-year gap Z-score, weekly or daily, with chart |
| DDM | Included whenever a dividend exists | Auto steady-payer rule (3 yrs paid, no cut > 25%, yield ≥ 2%, payout 30–90%); Yield group always; On/Off option |
| Minority interest | Ignored | Deducted in FCFF DCF, reverse DCF, EPV, EV bands, peer EV/EBITDA (yfinance field; 0 if missing) |
| Graham Number | TTM EPS | 3-year average fiscal-year EPS (TTM fallback) |
| Graham growth | Yield-scaled only (× 4.4 ÷ Y) | Book form by default; yield-scaled as an option |
| Free cash flow | yfinance, then CFO + CFI proxy | Screener's own "Free Cash Flow" row first (10+ years), then yfinance; proxy only as last resort, with a warning |

## Unchanged
Symbol picker, Secrets tab, Dhan integration, source toggles, editable inputs, auto assumptions (β, ERP by size,
mid-year discounting), DCF sensitivity, reverse DCF, residual income, peers, SOTP.

## Known limits
- Yearly share counts are Screener net profit ÷ EPS with a ±12% noise filter; older EVs and book values are approximate.
- Net-debt history uses Screener borrowings and yfinance cash (about 4 years); older years reuse the earliest cash figure.
- The history-based composite uses only methods with yearly history, so it differs from the headline composite and
  from the Pine Z.
- yfinance industry names for stocks outside the Nifty 500 are matched by keyword — check a few.

# v2.1 — Nifty 500 Z Scanner (28 Oct 2026)

New page **Nifty 500 Z Scanner** (Streamlit sidebar navigation; files `pages/2_Nifty_500_Z_Scanner.py`,
`scanner_core.py`, `zcore.py`).

- Scores all 500 stocks with the same history-based composite and weekly Z as the Workbench Summary tab
  (`zcore.py` is now the single engine for both), using each stock's valuation group.
- Signals: **Strong value** (weekly Z ≤ −1.5), **Value** (≤ −1), each with **· turning up** when weekly Z has
  risen ≥ 0.25 from its 4-week low and the close is above the 20-day average; Neutral / Rich / Very rich.
- Filters: size (Large = top 100 by market cap, Mid = next 150, Small), valuation group, signal, minimum methods.
  Table links straight into the Workbench (`/?s=SYMBOL`); CSV download; Z vs upside scatter.
- Data: yfinance batch prices (refresh after 12 h) + Screener pages (refresh after 7 days, 3 workers with a pause
  after each stock; ~1 s per stock in testing → first full scan ≈ 10–15 min including prices). Cached in
  `.scan_cache/`. "Re-score from cached data" applies new settings without downloading.
- If Screener's consolidated page has < 6 fiscal years (e.g. HDFC AMC), the scanner uses the standalone page when it
  is longer; the Workbench shows a hint to switch off "Prefer consolidated".

Scanner limits: one Ke / WACC / growth setting for all stocks; net debt = Screener borrowings (no cash) and no
minority deduction in the scan (applied to past and present EVs alike); on Streamlit Community Cloud the cache is
wiped when the app restarts, so the first scan after a restart downloads everything again.
