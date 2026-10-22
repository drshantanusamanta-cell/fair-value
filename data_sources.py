"""Data layer: yfinance, Screener.in and the DhanHQ Data API (company info).

Every fetcher returns a plain dict and never raises — failures come back as {"ok": False, "error": ...}
so the app can keep working on whatever sources did respond. Amounts are converted to absolute rupees
(Screener reports ₹ crore; 1 Cr = 1e7).
"""
from __future__ import annotations

import io
import math
import re
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

CR = 1e7
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/126.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9",
}
DHAN_BASE = "https://api.dhan.co/v2"
DHAN_SCRIP_MASTER = "https://images.dhan.co/api-data/api-scrip-master.csv"


# ───────────────────────────── helpers ─────────────────────────────
def num(x: Any) -> float:
    """Parse '1,234.5', '12.3%', '₹ 45', 7, None → float (nan when unparseable)."""
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).replace(",", "").replace("₹", "").replace("%", "").replace("Cr.", "").replace("Cr", "").strip()
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else float("nan")


def finite(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and math.isfinite(float(x))


def _clean_label(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ").replace("+", "")).strip()


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def period_to_date(label: str) -> pd.Timestamp | None:
    """'Mar 2024' → 2024-03-31. 'TTM' → None."""
    try:
        return pd.to_datetime(label, format="%b %Y") + pd.offsets.MonthEnd(0)
    except Exception:
        return None


# ───────────────────────────── Screener.in ─────────────────────────────
def _parse_data_table(table) -> pd.DataFrame | None:
    if table is None:
        return None
    head = table.find("thead")
    headers = [th.get_text(" ", strip=True) for th in (head.find_all("th") if head else [])]
    rows = []
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        vals = [c.get_text(" ", strip=True) for c in cells]
        rows.append(vals)
    if not rows:
        return None
    width = max(len(r) for r in rows)
    if len(headers) < width:
        headers = headers + [f"col{i}" for i in range(len(headers), width)]
    df = pd.DataFrame([r + [""] * (width - len(r)) for r in rows], columns=headers[:width])
    return df


def _statement(soup, section_id: str) -> pd.DataFrame | None:
    sec = soup.find("section", id=section_id)
    if sec is None:
        return None
    df = _parse_data_table(sec.find("table", class_="data-table"))
    if df is None or df.empty:
        return None
    label_col = df.columns[0]
    df[label_col] = df[label_col].map(_clean_label)
    df = df.set_index(label_col)
    df.index.name = "Item"
    df = df.apply(lambda col: col.map(num))
    return df


def screener_row(df: pd.DataFrame | None, *cands: str) -> pd.Series | None:
    if df is None:
        return None
    for c in cands:
        for idx in df.index:
            if idx.lower().startswith(c.lower()):
                return df.loc[idx]
    return None


def fetch_screener(symbol: str, prefer_consolidated: bool = True, timeout: int = 20) -> dict:
    out: dict[str, Any] = {"ok": False, "error": None, "url": None, "basis": None, "ratios": {}, "ratios_raw": {},
                           "pl": None, "bs": None, "cf": None, "shp": None, "peers": None, "about": None,
                           "warehouse_id": None}
    sess = requests.Session()
    sess.headers.update(UA)
    variants = [("consolidated/", "Consolidated"), ("", "Standalone")]
    if not prefer_consolidated:
        variants.reverse()
    last_err = None
    for path, basis in variants:
        url = f"https://www.screener.in/company/{symbol.upper()}/{path}"
        try:
            r = sess.get(url, timeout=timeout)
        except Exception as e:  # network error
            last_err = f"{type(e).__name__}: {e}"
            continue
        if r.status_code != 200:
            last_err = f"HTTP {r.status_code} for {url}"
            continue
        soup = BeautifulSoup(r.text, "lxml")
        pl = _statement(soup, "profit-loss")
        if pl is None or pl.dropna(how="all").empty or pl.notna().sum().sum() < 5:
            last_err = f"No P&L data at {url}"
            continue
        out.update(ok=True, url=url, basis=basis, pl=pl,
                   bs=_statement(soup, "balance-sheet"), cf=_statement(soup, "cash-flow"),
                   shp=_statement(soup, "shareholding"))
        for li in soup.select("#top-ratios li"):
            name = li.select_one(".name")
            if not name:
                continue
            key = name.get_text(" ", strip=True)
            nums = [n.get_text(strip=True) for n in li.select(".number")]
            out["ratios_raw"][key] = li.select_one(".value").get_text(" ", strip=True) if li.select_one(".value") else " / ".join(nums)
            out["ratios"][key] = num(nums[0]) if nums else float("nan")
        about = soup.select_one(".company-profile .about p") or soup.select_one(".about p")
        out["about"] = about.get_text(" ", strip=True) if about else None
        info = soup.select_one("#company-info")
        if info is not None:
            out["warehouse_id"] = info.get("data-warehouse-id")
        if out["warehouse_id"]:
            try:
                pr = sess.get(f"https://www.screener.in/api/company/{out['warehouse_id']}/peers/", timeout=timeout)
                if pr.status_code == 200:
                    psoup = BeautifulSoup(pr.text, "lxml")
                    peers = _parse_data_table(psoup.find("table"))
                    if peers is not None and not peers.empty:
                        out["peers"] = peers
            except Exception:
                pass
        return out
    out["error"] = last_err or "Screener returned no data"
    return out


# ───────────────────────────── yfinance ─────────────────────────────
def fetch_yfinance(symbol: str, history_period: str = "15y") -> dict:
    out: dict[str, Any] = {"ok": False, "error": None, "info": {}, "history": None, "financials": None,
                           "balance_sheet": None, "cashflow": None, "targets": None, "eps_estimate": None}
    try:
        import yfinance as yf
    except Exception as e:
        out["error"] = f"yfinance not installed: {e}"
        return out
    t = yf.Ticker(f"{symbol.upper()}.NS")
    errors = []
    for attr, key in [("info", "info"), ("financials", "financials"), ("balance_sheet", "balance_sheet"),
                      ("cashflow", "cashflow"), ("analyst_price_targets", "targets"), ("earnings_estimate", "eps_estimate")]:
        try:
            val = getattr(t, attr)
            out[key] = val if val is not None else out[key]
        except Exception as e:
            errors.append(f"{attr}: {type(e).__name__}")
    try:
        h = t.history(period=history_period, auto_adjust=False)
        if h is not None and not h.empty:
            h.index = pd.to_datetime(h.index).tz_localize(None)
            out["history"] = h
    except Exception as e:
        errors.append(f"history: {type(e).__name__}")
    out["ok"] = bool(out["info"]) or out["history"] is not None
    out["error"] = "; ".join(errors) if errors else None
    return out


def yf_row(df: pd.DataFrame | None, *names: str) -> pd.Series | None:
    """Row from a yfinance statement (columns = period dates), returned oldest → newest."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    return None


# ───────────────────────────── DhanHQ ─────────────────────────────
def load_dhan_scrip_master(url: str = DHAN_SCRIP_MASTER, timeout: int = 60) -> pd.DataFrame | None:
    """NSE cash-equity rows of Dhan's instrument master with normalised columns: symbol, security_id, name, series."""
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        r.raise_for_status()
        raw = r.content
    except Exception:
        return None
    head = pd.read_csv(io.BytesIO(raw), nrows=0).columns
    if "SEM_SMST_SECURITY_ID" in head:  # compact file
        cols = {"SEM_EXM_EXCH_ID": "exch", "SEM_SEGMENT": "segment", "SEM_SMST_SECURITY_ID": "security_id",
                "SEM_INSTRUMENT_NAME": "instrument", "SEM_TRADING_SYMBOL": "symbol", "SEM_SERIES": "series",
                "SM_SYMBOL_NAME": "name"}
    else:  # detailed file
        cols = {"EXCH_ID": "exch", "SEGMENT": "segment", "SECURITY_ID": "security_id", "INSTRUMENT": "instrument",
                "UNDERLYING_SYMBOL": "symbol", "SERIES": "series", "SYMBOL_NAME": "name"}
    use = [c for c in cols if c in head]
    df = pd.read_csv(io.BytesIO(raw), usecols=use, dtype=str, low_memory=False).rename(columns=cols)
    mask = df["exch"].str.upper().eq("NSE")
    if "segment" in df:
        mask &= df["segment"].str.upper().isin(["E", "NSE_EQ"])
    if "instrument" in df:
        mask &= df["instrument"].str.upper().eq("EQUITY")
    return df[mask].reset_index(drop=True)


def dhan_security_id(master: pd.DataFrame | None, symbol: str) -> tuple[str | None, str | None]:
    if master is None or master.empty:
        return None, None
    sym = symbol.upper().strip()
    hit = master[master["symbol"].str.upper() == sym]
    if "series" in hit and len(hit) > 1:
        eq = hit[hit["series"].str.upper() == "EQ"]
        hit = eq if not eq.empty else hit
    if hit.empty:
        return None, None
    row = hit.iloc[0]
    return str(row["security_id"]), (row.get("name") if isinstance(row.get("name"), str) else None)


def fetch_dhan_company_info(security_id: str, access_token: str, client_id: str,
                            sections: tuple[str, ...] = ("CO", "RATIOS", "SHP"), timeout: int = 20) -> dict:
    """POST /v2/data/companyinfo. Needs an active Dhan Data API subscription.

    Tries all sections in one call; if that is refused, retries one section at a time and merges.
    """
    out: dict[str, Any] = {"ok": False, "error": None, "status": None, "data": {}, "raw": None}
    if not (security_id and access_token):
        out["error"] = "Missing security ID or access token"
        return out
    headers = {"access-token": access_token.strip(), "client-id": (client_id or "").strip(),
               "Content-Type": "application/json", "Accept": "application/json"}

    def call(metrics):
        body = {"securityId": str(security_id), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY", "metrics": list(metrics)}
        r = requests.post(f"{DHAN_BASE}/data/companyinfo", json=body, headers=headers, timeout=timeout)
        try:
            payload = r.json()
        except Exception:
            payload = {"text": r.text[:500]}
        return r.status_code, payload

    try:
        status, payload = call(sections)
        out["status"], out["raw"] = status, payload
        if status == 200 and isinstance(payload, dict):
            out["data"] = payload.get("data", payload)
            out["ok"] = bool(out["data"])
            return out
        merged, errs = {}, [f"combined: HTTP {status} {_dhan_err(payload)}"]
        for s in sections:
            st, pl = call((s,))
            if st == 200 and isinstance(pl, dict):
                d = pl.get("data", pl)
                merged.update(d if isinstance(d, dict) else {s: d})
            else:
                errs.append(f"{s}: HTTP {st} {_dhan_err(pl)}")
        out["data"], out["ok"] = merged, bool(merged)
        out["error"] = None if merged else "; ".join(errs)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _dhan_err(payload) -> str:
    if isinstance(payload, dict):
        return str(payload.get("errorMessage") or payload.get("message") or payload.get("errorCode") or "")[:160]
    return ""


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON. Lists of {name/label: …, value: …} pairs become name-keyed entries."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        name_k = next((k for k in obj if _norm_key(k) in ("name", "label", "title", "key", "metric", "parameter")), None)
        val_k = next((k for k in obj if _norm_key(k) in ("value", "val", "data", "amount")), None)
        if name_k and val_k and not isinstance(obj[val_k], (dict, list)) and len(obj) <= 4:
            out[f"{prefix}.{obj[name_k]}" if prefix else str(obj[name_k])] = obj[val_k]
            return out
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if all(not isinstance(v, (dict, list)) for v in obj):
            out[prefix] = ", ".join(str(v) for v in obj)
        else:
            for i, v in enumerate(obj):
                out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


# Synonyms matched against the normalised last segment of each flattened Dhan key.
DHAN_SYNONYMS = {
    "pe": ["pe", "peratio", "ttmpe", "stockpe", "priceearnings", "pettm"],
    "pb": ["pb", "pbratio", "pricetobook", "pricetobookvalue", "ptb"],
    "eps": ["eps", "ttmeps", "epsttm", "earningspershare", "dilutedeps"],
    "bvps": ["bookvalue", "bookvaluepershare", "bvps", "bv"],
    "roe": ["roe", "returnonequity"],
    "roce": ["roce", "returnoncapitalemployed"],
    "div_yield": ["dividendyield", "divyield", "dy"],
    "market_cap": ["marketcap", "mcap", "marketcapitalisation", "marketcapitalization"],
    "sector_pe": ["sectorpe", "industrype", "indpe", "sectorpratio"],
    "debt_equity": ["debttoequity", "de", "debtequity"],
    "face_value": ["facevalue", "fv"],
    "sector": ["sector", "sectorname"],
    "industry": ["industry", "industryname"],
    "promoter": ["promoter", "promoters", "promoterholding"],
    "fii": ["fii", "fiis", "foreigninstitutions"],
    "dii": ["dii", "diis", "domesticinstitutions"],
}


def dhan_extract(flat: dict[str, Any]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for full_key, value in flat.items():
        last = _norm_key(re.split(r"[.\[]", full_key)[-1].rstrip("]"))
        for field, syns in DHAN_SYNONYMS.items():
            if field in found:
                continue
            if last in syns:
                found[field] = value if field in ("sector", "industry") else num(value)
                found[f"{field}__key"] = full_key
    return found


# ───────────────────────────── Merge ─────────────────────────────
def _last(s: pd.Series | None, skip_ttm: bool = False) -> float:
    if s is None:
        return float("nan")
    s2 = s.drop(labels=[c for c in s.index if skip_ttm and str(c).upper() == "TTM"], errors="ignore").dropna()
    return float(s2.iloc[-1]) if not s2.empty else float("nan")


def _annual(s: pd.Series | None) -> pd.Series | None:
    """Screener row → Series indexed by fiscal year-end date (TTM dropped)."""
    if s is None:
        return None
    idx, vals = [], []
    for k, v in s.items():
        d = period_to_date(str(k))
        if d is not None and finite(v):
            idx.append(d)
            vals.append(float(v))
    return pd.Series(vals, index=idx).sort_index() if idx else None


def build_fundamentals(yfd: dict, scr: dict, dh: dict) -> tuple[dict, pd.DataFrame, dict]:
    """Merge the three sources. Returns (values, provenance table, extras)."""
    info = yfd.get("info") or {}
    hist = yfd.get("history")
    fin, bal, cfs = yfd.get("financials"), yfd.get("balance_sheet"), yfd.get("cashflow")
    rat = scr.get("ratios") or {}
    pl, bs, cf = scr.get("pl"), scr.get("bs"), scr.get("cf")
    dx = dh.get("extract") or {}

    cand: dict[str, dict[str, float]] = {}

    def add(field, source, value):
        if finite(value):
            cand.setdefault(field, {})[source] = float(value)

    # Price & size
    last_close = float(hist["Close"].dropna().iloc[-1]) if hist is not None and not hist.empty else float("nan")
    add("price", "yfinance", info.get("currentPrice") or info.get("regularMarketPrice") or last_close)
    add("price", "Screener", rat.get("Current Price"))
    add("shares", "yfinance", info.get("sharesOutstanding"))
    if finite(rat.get("Market Cap")) and finite(rat.get("Current Price")) and rat.get("Current Price"):
        add("shares", "Screener", rat["Market Cap"] * CR / rat["Current Price"])
    add("market_cap", "Screener", (rat.get("Market Cap") or float("nan")) * CR)
    add("market_cap", "yfinance", info.get("marketCap"))
    add("market_cap", "Dhan", dx.get("market_cap", float("nan")) * CR if finite(dx.get("market_cap")) and dx.get("market_cap") < 1e9 else dx.get("market_cap"))

    # Per-share and returns
    add("eps_ttm", "Screener", _last(screener_row(pl, "EPS in Rs", "EPS")))
    add("eps_ttm", "yfinance", info.get("trailingEps"))
    add("eps_ttm", "Dhan", dx.get("eps"))
    add("bvps", "Screener", rat.get("Book Value"))
    add("bvps", "yfinance", info.get("bookValue"))
    add("bvps", "Dhan", dx.get("bvps"))
    roe_scr = rat.get("ROE")
    add("roe", "Screener", roe_scr / 100 if finite(roe_scr) else None)
    add("roe", "yfinance", info.get("returnOnEquity"))
    add("roe", "Dhan", dx["roe"] / 100 if finite(dx.get("roe")) and abs(dx["roe"]) > 1.5 else dx.get("roe"))
    add("pe_reported", "Screener", rat.get("Stock P/E"))
    add("pe_reported", "yfinance", info.get("trailingPE"))
    add("pe_reported", "Dhan", dx.get("pe"))
    add("sector_pe", "Dhan", dx.get("sector_pe"))

    # Dividends
    eps_fy = _last(screener_row(pl, "EPS in Rs", "EPS"), skip_ttm=True)
    payout = _last(screener_row(pl, "Dividend Payout"), skip_ttm=True)
    if finite(eps_fy) and finite(payout):
        add("dps", "Screener", eps_fy * payout / 100)
        add("payout", "Screener", payout / 100)
    add("dps", "yfinance", info.get("dividendRate") or info.get("trailingAnnualDividendRate"))
    add("payout", "yfinance", info.get("payoutRatio"))

    # Operating items (absolute ₹)
    pbt = _last(screener_row(pl, "Profit before tax"))
    interest = _last(screener_row(pl, "Interest"))
    if finite(pbt) and finite(interest):
        add("ebit", "Screener", (pbt + interest) * CR)
    add("ebit", "yfinance", _last(yf_row(fin, "EBIT", "Operating Income")))
    add("interest", "Screener", interest * CR if finite(interest) else None)
    add("interest", "yfinance", _last(yf_row(fin, "Interest Expense")))
    tax_pct = _last(screener_row(pl, "Tax %"))
    add("tax_rate", "Screener", tax_pct / 100 if finite(tax_pct) else None)
    add("tax_rate", "yfinance", _last(yf_row(fin, "Tax Rate For Calcs")))
    add("revenue", "Screener", _last(screener_row(pl, "Sales", "Revenue")) * CR)
    add("revenue", "yfinance", info.get("totalRevenue"))
    add("ebitda", "Screener", _last(screener_row(pl, "Operating Profit", "Financing Profit")) * CR)
    add("ebitda", "yfinance", info.get("ebitda"))

    # Cash flows
    yf_fcf = yf_row(cfs, "Free Cash Flow")
    add("fcf", "yfinance", _last(yf_fcf))
    cfo_s, cfi_s = screener_row(cf, "Cash from Operating"), screener_row(cf, "Cash from Investing")
    if cfo_s is not None and cfi_s is not None:
        add("fcf", "Screener (CFO+CFI proxy)", (_last(cfo_s) + _last(cfi_s)) * CR)
    add("cfo", "Screener", _last(cfo_s) * CR)
    add("cfo", "yfinance", _last(yf_row(cfs, "Operating Cash Flow")))

    # Balance sheet
    add("debt", "yfinance", _last(yf_row(bal, "Total Debt")) if yf_row(bal, "Total Debt") is not None else info.get("totalDebt"))
    add("debt", "Screener", _last(screener_row(bs, "Borrowings", "Borrowing")) * CR)
    add("cash", "yfinance", _last(yf_row(bal, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"))
        if yf_row(bal, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents") is not None else info.get("totalCash"))
    add("beta", "yfinance", info.get("beta"))
    add("forward_eps", "yfinance", info.get("forwardEps"))
    add("target_mean", "yfinance", info.get("targetMeanPrice"))
    add("target_high", "yfinance", info.get("targetHighPrice"))
    add("target_low", "yfinance", info.get("targetLowPrice"))
    add("analysts", "yfinance", info.get("numberOfAnalystOpinions"))

    priority = {
        "default": ["Screener", "yfinance", "Dhan", "Screener (CFO+CFI proxy)"],
        "price": ["yfinance", "Screener"],
        "shares": ["Screener", "yfinance"],
        "fcf": ["yfinance", "Screener (CFO+CFI proxy)"],
        "debt": ["yfinance", "Screener"],
        "cash": ["yfinance"],
        "beta": ["yfinance"],
        "sector_pe": ["Dhan"],
    }
    labels = {
        "price": "Price (₹)", "shares": "Shares outstanding", "market_cap": "Market cap (₹)", "eps_ttm": "EPS TTM (₹)",
        "bvps": "Book value / share (₹)", "roe": "ROE", "pe_reported": "P/E (reported)", "sector_pe": "Sector P/E",
        "dps": "Dividend / share (₹)", "payout": "Payout ratio", "ebit": "EBIT TTM (₹)", "interest": "Interest (₹)",
        "tax_rate": "Tax rate", "revenue": "Revenue (₹)", "ebitda": "EBITDA / Op. profit (₹)", "fcf": "Free cash flow (₹)",
        "cfo": "Cash from operations (₹)", "debt": "Total debt (₹)", "cash": "Cash & equivalents (₹)", "beta": "Beta",
        "forward_eps": "Forward EPS (₹)", "target_mean": "Analyst target — mean (₹)", "target_high": "Analyst target — high (₹)",
        "target_low": "Analyst target — low (₹)", "analysts": "No. of analysts",
    }
    values, prov = {}, []
    for field, label in labels.items():
        srcs = cand.get(field, {})
        order = priority.get(field, priority["default"])
        chosen_src = next((s for s in order if s in srcs), next(iter(srcs), None))
        val = srcs.get(chosen_src, float("nan")) if chosen_src else float("nan")
        values[field] = val
        others = [v for s, v in srcs.items() if s != chosen_src]
        spread = max(abs(o / val - 1) for o in others) if others and finite(val) and val != 0 else float("nan")
        prov.append({"Field": label, "key": field, "Value": val, "Source": chosen_src or "—",
                     "yfinance": srcs.get("yfinance"), "Screener": srcs.get("Screener", srcs.get("Screener (CFO+CFI proxy)")),
                     "Dhan": srcs.get("Dhan"), "Max disagreement": spread})
    values["net_debt"] = (values["debt"] if finite(values["debt"]) else 0.0) - (values["cash"] if finite(values["cash"]) else 0.0)

    # Histories for growth & bands
    extras = {
        "eps_annual": _annual(screener_row(pl, "EPS in Rs", "EPS")),
        "sales_annual": _annual(screener_row(pl, "Sales", "Revenue")),
        "fcf_annual": yf_fcf,
        "cfo_annual": _annual(cfo_s),
        "name": info.get("longName") or info.get("shortName") or dh.get("name"),
        "sector": info.get("sector") or dx.get("sector"),
        "industry": info.get("industry") or dx.get("industry"),
        "summary": scr.get("about") or info.get("longBusinessSummary"),
    }
    if extras["eps_annual"] is None:
        e = yf_row(fin, "Diluted EPS", "Basic EPS")
        extras["eps_annual"] = e.dropna() if e is not None else None
    return values, pd.DataFrame(prov), extras
