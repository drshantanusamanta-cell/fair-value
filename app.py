"""Fair Value Workbench — multi-method intrinsic value for NSE stocks.

Sources: yfinance (prices, statements, estimates), Screener.in (10-yr Indian financials, ratios, peers,
shareholding) and the DhanHQ Data API company-info endpoint (overview, ratios, shareholding).
Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_sources as ds
import valuation as v

st.set_page_config(page_title="Fair Value Workbench", page_icon="⚖️", layout="wide")

finite = ds.finite


# ───────────────────────────── formatting ─────────────────────────────
def inr(x, d: int = 1) -> str:
    return f"₹{x:,.{d}f}" if finite(x) else "—"


def crore(x) -> str:
    return f"₹{x / 1e7:,.0f} Cr" if finite(x) else "—"


def pct(x, d: int = 1, sign: bool = False) -> str:
    if not finite(x):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def up(fv, price):
    return fv / price - 1 if finite(fv) and finite(price) and price else float("nan")


SECRETS_FILE = Path(__file__).parent / ".streamlit" / "secrets.toml"


SECRET_ALIASES = {
    "client_id": ["client_id", "clientid", "dhan_client_id", "dhanclientid"],
    "access_token": ["access_token", "accesstoken", "token", "api_token", "dhan_access_token", "dhan_token", "jwt"],
}


def _nk(k) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def secret(section: str, key: str) -> str:
    """Session override → st.secrets → environment variable.

    Tolerant of layout: [dhan] / [Dhan] / [DHAN] sections, key aliases (token, accessToken, …) and
    flat top-level keys such as DHAN_ACCESS_TOKEN or access_token.
    """
    override = st.session_state.get(f"_secret_{section}_{key}")
    if override:
        return str(override)
    names = SECRET_ALIASES.get(key, [key])
    want = {_nk(n) for n in names} | {_nk(f"{section}_{n}") for n in names}
    try:
        top = {k: st.secrets[k] for k in st.secrets.keys()}
        for k, val in top.items():  # [dhan] section, any case
            if hasattr(val, "keys") and _nk(k) == _nk(section):
                for kk in val.keys():
                    if _nk(kk) in want and str(val[kk]).strip():
                        return str(val[kk]).strip()
        for k, val in top.items():  # flat top-level keys
            if not hasattr(val, "keys") and _nk(k) in want and str(val).strip():
                return str(val).strip()
    except Exception as e:
        st.session_state["_secrets_error"] = f"{type(e).__name__}: {e}"
    for n in names:
        env = os.environ.get(f"{section}_{n}".upper()) or os.environ.get(n.upper())
        if env:
            return env.strip()
    return ""


def visible_secret_names() -> list[str]:
    """Secret key names only (never values), for diagnosing layout problems."""
    out = []
    try:
        for k in st.secrets.keys():
            val = st.secrets[k]
            out += [f"[{k}] {kk}" for kk in val.keys()] if hasattr(val, "keys") else [k]
    except Exception as e:
        out.append(f"⚠️ could not read secrets — {type(e).__name__}: {e}")
    return out


def save_secrets_section(section: str, values: dict, path: Path = SECRETS_FILE) -> None:
    """Rewrite only [section] in secrets.toml, keeping every other section intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept, skipping = [], False
    for line in lines:
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            skipping = s.strip("[] ") == section
        if not skipping:
            kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    block = [f"[{section}]"] + [f"{k} = {json.dumps(str(val))}" for k, val in values.items()]
    path.write_text("\n".join(kept + ([""] if kept else []) + block) + "\n", encoding="utf-8")


def mask(s: str) -> str:
    return "—" if not s else ("•" * 8 + s[-4:] if len(s) > 8 else "•" * len(s))


def jwt_expiry(token: str) -> datetime | None:
    """Dhan access tokens are JWTs — read the `exp` claim (no signature check, display only)."""
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return datetime.fromtimestamp(int(claims["exp"]))
    except Exception:
        return None


# ───────────────────────────── cached fetchers ─────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_yf(sym: str) -> dict:
    return ds.fetch_yfinance(sym)


@st.cache_data(ttl=3600, show_spinner=False)
def get_screener(sym: str, consolidated: bool) -> dict:
    return ds.fetch_screener(sym, consolidated)


class _NotCached(Exception):
    """Raised inside a cached function so Streamlit does NOT cache a failed fetch (exceptions are never cached)."""
    def __init__(self, result):
        super().__init__("fetch failed")
        self.result = result


@st.cache_data(ttl=86400, show_spinner=False)
def _nse_list_cached():
    """NSE equity symbols + names + Dhan security IDs from Dhan's public scrip master (no token needed)."""
    try:
        df = ds.load_nse_equities()
    except Exception:
        raise _NotCached(None)
    if df is None or df.empty:
        raise _NotCached(None)
    return df


def get_nse_list():
    try:
        return _nse_list_cached()
    except _NotCached:
        return None  # retried on the next run instead of being stuck for 24 h


def get_master():
    """Same list, in the column layout ds.dhan_security_id() expects (one download serves both uses)."""
    df = get_nse_list()
    if df is None:
        return None
    return pd.DataFrame({"symbol": df["SYMBOL"], "security_id": df["SID"], "name": df["NAME"], "series": "EQ"})


@st.cache_data(ttl=1800, show_spinner=False)
def _dhan_cached(sid: str, token: str, client_id: str) -> dict:
    r = ds.fetch_dhan_company_info(sid, token, client_id)
    if not r.get("ok"):
        raise _NotCached(r)
    return r


def get_dhan(sid: str, token: str, client_id: str) -> dict:
    try:
        return _dhan_cached(sid, token, client_id)
    except _NotCached as e:
        return e.result  # error shown now, retried on the next run


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _index_cached():
    try:
        import yfinance as yf
        h = yf.Ticker("^NSEI").history(period="3y", interval="1d")
    except Exception:
        raise _NotCached(None)
    if h is None or h.empty:
        raise _NotCached(None)
    c = h["Close"]
    c.index = pd.to_datetime(c.index).tz_localize(None)
    return c


def get_index_close():
    try:
        return _index_cached()
    except _NotCached:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_peer_info(sym: str) -> dict:
    try:
        import yfinance as yf
        return yf.Ticker(f"{sym}.NS").info or {}
    except Exception:
        return {}


# ───────────────────────────── sidebar ─────────────────────────────
with st.sidebar:
    st.header("⚖️ Fair Value Workbench")
    if "fv_symbol" not in st.session_state:
        st.session_state["fv_symbol"] = str(st.query_params.get("s", "ITC")).strip().upper()
    # Fixed widget key: passing value= would give the box a new identity whenever the symbol changes, and Streamlit
    # would then discard the next thing typed. A suggestion click sets _fv_input_pending, applied here before the
    # widget exists (a widget's state can't be changed after it is drawn in the same run).
    if "_fv_input_pending" in st.session_state:
        st.session_state["fv_input"] = st.session_state.pop("_fv_input_pending")
    st.session_state.setdefault("fv_input", st.session_state["fv_symbol"])
    with st.form("sym_form", border=False):
        typed = st.text_input("NSE symbol", key="fv_input",
                              placeholder="e.g. ITC, HDFCBANK, or a company name",
                              help="If the text is not an NSE symbol, similar symbols are suggested.")
        go_btn = st.form_submit_button("🔍 Go", type="primary", width="stretch")
    if go_btn and typed.strip():
        q = typed.strip().upper()
        q = q[:-3] if q.endswith(".NS") else q
        st.session_state.pop("fv_suggest", None)
        nse = get_nse_list() if ("." not in q and not q.startswith("^")) else None  # list unavailable → use as typed
        if nse is not None and q not in set(nse["SYMBOL"]):
            st.session_state["fv_suggest"] = (q, ds.suggest_symbols(q, nse))
        else:
            st.session_state["fv_symbol"] = q
    symbol = st.session_state["fv_symbol"]

    with st.expander("Data sources"):
        use_scr = st.toggle("Screener.in", value=True)
        consolidated = st.toggle("Prefer consolidated financials", value=True)
        use_yf = st.toggle("yfinance", value=True)
        use_dhan = st.toggle("Dhan company info (experimental)", value=True,
                             help="POST /v2/data/companyinfo is not in Dhan's official docs or SDK — it may not exist.")

    dhan_cid = secret("dhan", "client_id")
    dhan_tok = secret("dhan", "access_token")
    with st.expander("Dhan API", expanded=False):
        st.caption(f"Credentials: {'✅ set' if dhan_tok else '⚠️ not set'} — view or edit them in the **🔐 Secrets** tab.")
        sid_override = st.text_input("Security ID override (optional)", value="")
        st.caption("⚠️ Experimental: the company-info endpoint is not in Dhan's official docs; it may fail even "
                   "with valid credentials.")

    st.subheader("Assumptions")
    _rf_secret = ds.num(secret("app", "risk_free_rate") or secret("", "RISK_FREE_RATE"))
    st.session_state.setdefault("rf_pct", float(_rf_secret) if finite(_rf_secret) and 0 < _rf_secret < 20 else 7.45)
    rf = st.number_input("Risk-free rate — 10Y G-Sec %", key="rf_pct", step=0.05, format="%.2f",
                         help="Enter today's 10-year G-Sec yield. Set RISK_FREE_RATE in Secrets to change the default.") / 100
    assump_box = st.container()   # filled with per-stock auto assumptions once the data is in

if st.session_state.get("fv_suggest"):
    q, opts = st.session_state["fv_suggest"]
    if opts:
        st.warning(f"**“{q}”** is not an NSE symbol. Did you mean one of these?")
        cols = st.columns(2)
        for i, (sym_, nm_) in enumerate(opts):
            if cols[i % 2].button(f"{sym_} · {nm_.title()}" if nm_ else sym_, key=f"sugg_{sym_}", width="stretch"):
                st.session_state.pop("fv_suggest", None)
                st.session_state["fv_symbol"] = sym_
                st.session_state["_fv_input_pending"] = sym_   # show the chosen symbol in the box
                st.rerun()
    else:
        st.warning(f"**“{q}”** is not an NSE symbol and nothing similar was found. Try the company name, e.g. *tata motors*.")
    st.stop()
if not symbol:
    st.info("Enter an NSE symbol in the sidebar.")
    st.stop()
st.query_params["s"] = symbol

# ───────────────────────────── fetch ─────────────────────────────
with st.spinner(f"Fetching {symbol} from the enabled sources…"):
    yfd = get_yf(symbol) if use_yf else {"ok": False, "error": "disabled"}
    scr = get_screener(symbol, consolidated) if use_scr else {"ok": False, "error": "disabled"}
    dh: dict = {"ok": False, "error": "disabled"}
    if use_dhan:
        if not dhan_tok:
            dh = {"ok": False, "error": "No access token — add it in the 🔐 Secrets tab"}
        else:
            sid, dname = (sid_override.strip(), None) if sid_override.strip() else ds.dhan_security_id(get_master(), symbol)
            if not sid:
                dh = {"ok": False, "error": "Security ID not found in Dhan's instrument master — enter it manually"}
            else:
                dh = dict(get_dhan(sid, dhan_tok, dhan_cid))
                dh.update(security_id=sid, name=dname)
                if dh.get("ok"):
                    dh["flat"] = ds.flatten(dh["data"])
                    dh["extract"] = ds.dhan_extract(dh["flat"])

vals, prov, ext = ds.build_fundamentals(yfd, scr, dh)

# ───────────────────────────── header ─────────────────────────────
name = ext.get("name") or symbol
st.title(f"{name}")
meta = " · ".join(x for x in [f"NSE: {symbol}", ext.get("sector"), ext.get("industry"),
                              scr.get("basis") and f"{scr['basis']} financials"] if x)
st.caption(meta)

badge = {True: "🟢", False: "🔴"}
s1, s2, s3 = st.columns(3)
s1.markdown(f"{badge[bool(yfd.get('ok'))]} **yfinance**  \n<small>{yfd.get('error') or 'ok'}</small>", unsafe_allow_html=True)
s2.markdown(f"{badge[bool(scr.get('ok'))]} **Screener.in**  \n<small>{scr.get('error') or scr.get('url') or 'ok'}</small>", unsafe_allow_html=True)
s3.markdown(f"{badge[bool(dh.get('ok'))]} **Dhan company info** (experimental)  \n<small>{dh.get('error') or ('security ID ' + str(dh.get('security_id')))}</small>", unsafe_allow_html=True)

if not finite(vals.get("price")):
    st.error("No price available from any source — check the symbol or enable yfinance / Screener.")
    st.stop()

tabs = st.tabs(["📊 Summary", "🧾 Inputs & sources", "💵 DCF", "🔁 Reverse DCF", "🏦 DDM & RI",
                "📈 Multiples", "📐 Graham & EPV", "🧩 SOTP", "🏢 Company (Dhan)", "🔐 Secrets"])
results: dict[str, dict] = {}

# ───────────────────────────── Secrets ─────────────────────────────
with tabs[9]:
    st.subheader("Dhan API credentials")
    tok_src = ("this session" if st.session_state.get("_secret_dhan_access_token")
               else "secrets.toml / Streamlit Cloud secrets" if dhan_tok else "not set")
    exp = jwt_expiry(dhan_tok) if dhan_tok else None
    a, b, c = st.columns(3)
    a.metric("Client ID", dhan_cid or "—")
    b.metric("Access token", mask(dhan_tok), tok_src, delta_color="off")
    if exp:
        left = exp - datetime.now()
        c.metric("Token expires", f"{exp:%d %b %Y %H:%M}",
                 "EXPIRED" if left.total_seconds() <= 0 else f"in {left.days}d {left.seconds // 3600}h",
                 delta_color="inverse" if left.total_seconds() <= 0 else "off")
    else:
        c.metric("Token expires", "—")
    if not dhan_tok:
        seen = visible_secret_names()
        st.error("No Dhan access token found. Secret names this app can see: "
                 + (", ".join(f"`{s}`" for s in seen) if seen else "**none** — the Secrets box is empty or hasn't "
                    "reached the app yet (save it, wait ~1 min or Reboot app)."))
        st.caption("Expected layout — a `[dhan]` section with `client_id` and `access_token` (flat keys like "
                   "`DHAN_ACCESS_TOKEN` also work). Use straight quotes `\"…\"`, not curly “…”.")

    with st.form("dhan_secrets_form"):
        new_cid = st.text_input("Client ID", value=dhan_cid)
        new_tok = st.text_input("Access token", value="", type="password",
                                placeholder="Leave blank to keep the current token")
        persist = st.checkbox(f"Save to `{SECRETS_FILE.relative_to(Path(__file__).parent)}`", value=True,
                              help="Unticked = use for this browser session only.")
        save = st.form_submit_button("💾 Save credentials", type="primary")
    if save:
        tok_final = new_tok.strip() or dhan_tok
        if not tok_final:
            st.error("Enter an access token.")
        else:
            st.session_state["_secret_dhan_client_id"] = new_cid.strip()
            st.session_state["_secret_dhan_access_token"] = tok_final
            if persist:
                try:
                    save_secrets_section("dhan", {"client_id": new_cid.strip(), "access_token": tok_final})
                    st.session_state["_secrets_msg"] = f"Saved to {SECRETS_FILE}"
                except Exception as e:
                    st.session_state["_secrets_msg"] = f"Using for this session only — could not write the file ({e})"
            else:
                st.session_state["_secrets_msg"] = "Using for this session only"
            st.rerun()
    if st.session_state.get("_secrets_msg"):
        st.success(st.session_state.pop("_secrets_msg"))
    if st.session_state.get("_secret_dhan_access_token") and st.button("↩️ Discard session override"):
        st.session_state.pop("_secret_dhan_access_token", None)
        st.session_state.pop("_secret_dhan_client_id", None)
        st.rerun()

    st.caption("**Streamlit Community Cloud:** the app's disk is wiped on reboot, so a token saved here lasts only "
               "until the app restarts. For a permanent change, paste this into **App settings → Secrets**:")
    st.code(f'[dhan]\nclient_id = "{dhan_cid or "YOUR_CLIENT_ID"}"\naccess_token = "PASTE_YOUR_TOKEN_HERE"', language="toml")
    st.caption("Dhan access tokens expire after about 24 h — when the expiry above turns red, generate a new one "
               "at web.dhan.co → Profile → DhanHQ Trading APIs and save it here. The token is never shown in full.")

# ───────────────────────────── Inputs & sources ─────────────────────────────
CR_FIELDS = {"market_cap", "ebit", "interest", "revenue", "ebitda", "fcf", "cfo", "debt", "cash", "shares"}
PCT_FIELDS = {"roe", "tax_rate", "payout"}


def to_display(key, x):
    if not finite(x):
        return None
    return x / 1e7 if key in CR_FIELDS else x * 100 if key in PCT_FIELDS else x


def from_display(key, x):
    if x is None or not finite(x):
        return float("nan")
    return x * 1e7 if key in CR_FIELDS else x / 100 if key in PCT_FIELDS else x


with tabs[1]:
    st.subheader("Inputs used by every model")
    st.caption("Edit **Value** to override any input — every tab recalculates. Amounts marked (Cr) are ₹ crore; "
               "shares are in crore; ratios in %. 'Max disagreement' is the largest gap between sources.")
    disp = prov.copy()
    unit = disp["key"].map(lambda k: " (Cr)" if k in CR_FIELDS else " (%)" if k in PCT_FIELDS else "")
    disp["Field"] = disp["Field"].str.replace(" (₹)", "", regex=False) + unit
    for col in ["Value", "yfinance", "Screener", "Dhan"]:
        disp[col] = pd.to_numeric(pd.Series([to_display(k, x) for k, x in zip(disp["key"], disp[col])],
                                            index=disp.index, dtype="object"), errors="coerce")
    disp["Max disagreement"] = pd.to_numeric(disp["Max disagreement"], errors="coerce") * 100
    edited = st.data_editor(
        disp.drop(columns=["key"]), key=f"inputs_{symbol}", hide_index=True, width="stretch",
        disabled=["Field", "Source", "yfinance", "Screener", "Dhan", "Max disagreement"],
        column_config={
            "Value": st.column_config.NumberColumn("Value", format="%.2f"),
            "yfinance": st.column_config.NumberColumn(format="%.2f"),
            "Screener": st.column_config.NumberColumn(format="%.2f"),
            "Dhan": st.column_config.NumberColumn(format="%.2f"),
            "Max disagreement": st.column_config.NumberColumn("Max disagreement %", format="%.1f"),
        })
    F = dict(vals)
    for k, x in zip(disp["key"], edited["Value"]):
        F[k] = from_display(k, x)
    F["net_debt"] = (F["debt"] if finite(F["debt"]) else 0.0) - (F["cash"] if finite(F["cash"]) else 0.0)
    flagged = prov[(prov["Max disagreement"] > 0.15)]
    if not flagged.empty:
        st.warning("Sources disagree by >15% on: " + ", ".join(flagged["Field"]) + ". Check these before trusting the output.")

# ───────────────────────────── derived inputs ─────────────────────────────
price, shares = F["price"], F["shares"]
if not finite(shares) and finite(F["market_cap"]):
    shares = F["market_cap"] / price
mcap = price * shares if finite(shares) else F["market_cap"]
net_debt = F["net_debt"]
sect_txt = f"{ext.get('sector') or ''} {ext.get('industry') or ''} {name}".lower()
is_fin = any(w in sect_txt for w in ["bank", "financial", "insurance", "nbfc", "finance", "capital markets"])

eps_annual, sales_annual = ext.get("eps_annual"), ext.get("sales_annual")
eps_cagr, eps_n = v.series_cagr(list(eps_annual.values) if eps_annual is not None else [], 5)
sales_cagr, sales_n = v.series_cagr(list(sales_annual.values) if sales_annual is not None else [], 5)
fcf_hist = ext.get("fcf_annual")
fwd_growth = (F["forward_eps"] / F["eps_ttm"] - 1) if finite(F["forward_eps"]) and finite(F["eps_ttm"]) and F["eps_ttm"] > 0 else None
hist_growth = next((g for g in [eps_cagr, sales_cagr, fwd_growth] if g is not None and finite(g)), None)
default_g1 = float(min(max(hist_growth if hist_growth is not None else 0.10, 0.0), 0.25))

# ── per-stock auto assumptions (editable; edits are remembered per stock for this session) ──
_hist = yfd.get("history")
_idx = get_index_close()
beta_reg, beta_obs = (v.regression_beta(_hist["Close"], _idx) if _hist is not None and _idx is not None else (None, 0))
AUTO = v.auto_assumptions(rf, mcap, hist_growth, beta_reg, F["roe"] if finite(F["roe"]) else None)
AN = AUTO["notes"]


# Overrides live in a plain dict: Streamlit deletes a widget's own state when it is not drawn (e.g. while you
# look at another stock), so widget keys alone would forget your edits.
OVR = st.session_state.setdefault("fv_overrides", {})


def _store(name: str, key: str, sym: str = symbol):
    OVR.setdefault(sym, {})[name] = st.session_state[key]


def aval(name: str, auto):
    return OVR.get(symbol, {}).get(name, auto)


def _reset_auto(sym=symbol):
    OVR.pop(sym, None)
    for k in [k for k in st.session_state.keys() if str(k).endswith(f"__{sym}")]:
        del st.session_state[k]


def kw(name: str) -> dict:
    """Common widget kwargs: per-stock key + store edits as overrides."""
    k = f"{name}__{symbol}"
    return {"key": k, "on_change": _store, "args": (name, k)}


with assump_box:
    st.caption(f"Auto-set for **{symbol}** · {AUTO['size']} · edit any value to override")
    erp = st.number_input("Equity risk premium %", value=float(aval("erp", AUTO["erp"] * 100)), step=0.25, format="%.2f",
                          help=AN["erp"], **kw("erp")) / 100
    beta = st.number_input("Beta", value=float(aval("beta", AUTO["beta"])), step=0.05, format="%.2f", min_value=0.1,
                           help=AN["beta"], **kw("beta"))
    g_term = st.number_input("Terminal growth %", value=float(aval("gterm", AUTO["g_term"] * 100)), step=0.25, format="%.2f",
                             help=AN["g_term"], **kw("gterm")) / 100
    years = st.slider("Explicit forecast years", 5, 15, int(aval("years", AUTO["years"])), help=AN["years"], **kw("years"))
    fade = st.toggle("Fade growth linearly to terminal", value=bool(aval("fade", AUTO["fade"])), **kw("fade"))
    mid_year = st.toggle("Mid-year discounting", value=bool(aval("mid", AUTO["mid_year"])), **kw("mid"))
    mos = st.slider("Required margin of safety %", 0, 50, int(aval("mos", AUTO["mos"] * 100)), help=AN["mos"], **kw("mos")) / 100
    if OVR.get(symbol):
        st.caption("✏️ Overridden here: " + ", ".join(OVR[symbol]))
    with st.expander("Why these values?"):
        for lab, k in [("ERP", "erp"), ("Beta", "beta"), ("Terminal growth", "g_term"), ("Forecast years", "years"),
                       ("Margin of safety", "mos"), ("RI terminal spread", "ri_spread")]:
            st.caption(f"**{lab}:** {AN[k]}")
        st.caption("Rules of thumb, not market-derived optimum values — see the README.")
    st.button("↺ Reset to auto values", on_click=_reset_auto, width="stretch")

ke = v.capm(rf, beta, erp)
tax = min(max(F["tax_rate"], 0.0), 0.45) if finite(F["tax_rate"]) else 0.2517
kd = (min(max(F["interest"] / F["debt"], 0.05), 0.18)
      if finite(F["interest"]) and finite(F["debt"]) and F["debt"] > 0 and F["interest"] > 0 else rf + 0.02)
wacc_ = v.wacc(ke, kd, tax, mcap, F["debt"] if finite(F["debt"]) else 0.0)

# ───────────────────────────── DCF ─────────────────────────────
with tabs[2]:
    st.subheader("Two-stage FCFF DCF")
    if is_fin:
        st.warning("This looks like a financial company. Free cash flow, EBIT and net debt are not meaningful for "
                   "lenders/insurers — lean on Residual income, Justified P/B and the P/E band instead.")
    c1, c2, c3 = st.columns(3)
    latest_fcf = F["fcf"]
    avg3 = float(fcf_hist.dropna().iloc[-3:].mean()) if fcf_hist is not None and len(fcf_hist.dropna()) >= 2 else latest_fcf
    base_choice = c1.radio("Base FCF", ["Latest FY", "3-yr average", "Custom"], index=0)
    if base_choice == "Custom":
        base_fcf = c1.number_input("Custom base FCF (₹ Cr)", value=float(latest_fcf / 1e7) if finite(latest_fcf) else 0.0) * 1e7
    else:
        base_fcf = latest_fcf if base_choice == "Latest FY" else avg3
    c1.caption(f"Latest {crore(latest_fcf)} · 3-yr avg {crore(avg3)}")

    g1 = c2.slider("Stage-1 growth % (year 1)", -10.0, 40.0, round(default_g1 * 100 * 2) / 2, 0.5) / 100
    c2.caption(f"EPS CAGR {pct(eps_cagr)} ({eps_n}y) · Sales CAGR {pct(sales_cagr)} ({sales_n}y) · "
               f"Fwd EPS growth {pct(fwd_growth)}")
    disc_basis = c3.radio("Discount at", ["WACC (FCFF)", "Cost of equity"], index=0)
    rate = c3.number_input("Discount rate %", value=round((wacc_ if disc_basis.startswith("WACC") else ke) * 100, 2),
                           step=0.25, format="%.2f", key=f"rate_{disc_basis}_{symbol}_{round(wacc_ if disc_basis.startswith('WACC') else ke, 4)}") / 100
    c3.caption(f"Ke {pct(ke)} = {pct(rf)} + β {beta:.2f} × {pct(erp)} · Kd {pct(kd)} · WACC {pct(wacc_)}")

    path_df = pd.DataFrame({"Year": range(1, years + 1), "Growth %": [x * 100 for x in v.growth_path(g1, g_term, years, fade)]})
    with st.expander("Edit year-by-year growth path"):
        path_df = st.data_editor(path_df, key=f"path_{symbol}_{g1}_{years}_{fade}_{g_term}", hide_index=True,
                                 disabled=["Year"], width="stretch")
    path = [float(x) / 100 for x in path_df["Growth %"].fillna(0)]

    dcf_rate = rate
    # WACC → FCFF valued as enterprise, net debt subtracted. Ke → FCFE ≈ FCFF − after-tax interest
    # (assumes zero net new borrowing), equity valued directly, so net debt is NOT subtracted again.
    use_fcfe = not disc_basis.startswith("WACC")
    int_at = F["interest"] * (1 - tax) if finite(F["interest"]) else 0.0
    cf0 = (base_fcf - int_at) if use_fcfe and finite(base_fcf) else base_fcf
    nd_used = 0.0 if use_fcfe else net_debt
    dcf_key = "DCF (FCFE)" if use_fcfe else "DCF (FCFF)"
    if use_fcfe:
        st.caption(f"FCFE basis: base FCF {crore(base_fcf)} − after-tax interest {crore(int_at)} = {crore(cf0)}; "
                   "net debt is not subtracted (equity is valued directly).")
    if not (finite(cf0) and cf0 > 0 and finite(shares)):
        st.info("Base FCF is missing or ≤ 0 — DCF not applicable. Try the 3-yr average, a custom base, or EPV.")
    elif rate <= g_term:
        st.error("Discount rate must exceed terminal growth.")
    else:
        res = v.dcf(cf0, path, rate, g_term, nd_used, shares, mid_year)
        ps = res["per_share"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fair value / share", inr(ps), pct(up(ps, price), sign=True))
        m2.metric("Enterprise value", crore(res["enterprise_value"]))
        m3.metric("Equity value", crore(res["equity_value"]))
        m4.metric("Terminal value share of EV", pct(res["tv_share"], 0))
        if res["tv_share"] > 0.75:
            st.warning("More than 75% of value sits in the terminal value — the result is very sensitive to "
                       "terminal growth and discount rate.")
        bridge = pd.DataFrame({
            "Item": ["PV of explicit FCF", "PV of terminal value", "Value before debt adj.", "less: net debt", "Equity value",
                     "÷ shares", "Fair value / share"],
            "Value": [crore(res["pv_explicit"]), crore(res["pv_terminal"]), crore(res["enterprise_value"]),
                      crore(nd_used), crore(res["equity_value"]), f"{shares / 1e7:,.2f} Cr", inr(ps)]})
        b1, b2 = st.columns([1, 1.4])
        b1.dataframe(bridge, hide_index=True, width="stretch")
        tbl = res["table"].copy()
        tbl["Growth"] = tbl["Growth"].map(lambda x: pct(x))
        tbl["Cash flow"] = tbl["Cash flow"].map(crore)
        tbl["PV"] = tbl["PV"].map(crore)
        tbl["Discount factor"] = tbl["Discount factor"].map(lambda x: f"{x:.3f}")
        b2.dataframe(tbl, hide_index=True, width="stretch", height=250)

        rates = [rate + d for d in (-0.02, -0.01, 0, 0.01, 0.02)]
        gts = [g_term + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
        sens = v.dcf_sensitivity(cf0, g1, years, fade, rates, gts, nd_used, shares, mid_year)
        fig = go.Figure(go.Heatmap(z=sens.values, x=sens.columns, y=sens.index, colorscale="RdYlGn",
                                   zmid=price, text=[[inr(z, 0) for z in row] for row in sens.values],
                                   texttemplate="%{text}", hovertemplate="r %{y} · g∞ %{x}: %{text}<extra></extra>"))
        fig.update_layout(title="Sensitivity — fair value per share (green = above price)", height=360,
                          xaxis_title="Terminal growth", yaxis_title="Discount rate", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        lo_v = sens.iloc[3, 1] if finite(sens.iloc[3, 1]) else ps
        hi_v = sens.iloc[1, 3] if finite(sens.iloc[1, 3]) else ps
        if ps > 0:
            results[dcf_key] = {"base": ps, "low": min(lo_v, hi_v), "high": max(lo_v, hi_v),
                                     "note": f"r {pct(rate)}, g1 {pct(g1)}, g∞ {pct(g_term)}"}

# ───────────────────────────── Reverse DCF ─────────────────────────────
with tabs[3]:
    st.subheader("Reverse DCF — what growth is the price assuming?")
    if not (finite(cf0) and cf0 > 0 and finite(shares)):
        st.info("Needs a positive base FCF (set on the DCF tab).")
    else:
        g_imp = v.reverse_dcf(price, shares, nd_used, cf0, years, dcf_rate, g_term, fade, mid_year)
        if g_imp is None:
            st.info("No solution between −50% and +150% stage-1 growth with these inputs.")
        else:
            st.metric(f"Implied stage-1 FCF growth ({years} yrs, {'fading' if fade else 'constant'})", pct(g_imp))
            comp_df = pd.DataFrame({
                "Benchmark": ["Price-implied", "Your stage-1 growth", f"EPS CAGR ({eps_n}y)", f"Sales CAGR ({sales_n}y)", "Forward EPS growth"],
                "Growth": [g_imp, g1, eps_cagr, sales_cagr, fwd_growth]}).dropna()
            fig = go.Figure(go.Bar(x=comp_df["Growth"] * 100, y=comp_df["Benchmark"], orientation="h",
                                   marker_color=["#F57C00"] + ["#78909C"] * (len(comp_df) - 1),
                                   text=[pct(x) for x in comp_df["Growth"]], textposition="outside"))
            fig.update_layout(height=280, xaxis_title="%", margin=dict(l=10, r=40, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            hist_ref = next((g for g in [eps_cagr, sales_cagr] if g is not None), None)
            if hist_ref is not None:
                if g_imp > hist_ref + 0.03:
                    st.warning(f"The price needs {pct(g_imp)} FCF growth — above the company's own record ({pct(hist_ref)}). "
                               "The market is paying for an acceleration.")
                elif g_imp < hist_ref - 0.03:
                    st.success(f"The price needs only {pct(g_imp)} growth — below the historical {pct(hist_ref)}. "
                               "Expectations look undemanding.")
                else:
                    st.info(f"Implied growth {pct(g_imp)} is in line with history ({pct(hist_ref)}).")
            st.caption(f"Uses the DCF tab's base cash flow ({crore(cf0)}, {dcf_key[5:9]}), discount rate {pct(dcf_rate)} and terminal growth {pct(g_term)}.")

# ───────────────────────────── DDM & RI ─────────────────────────────
with tabs[4]:
    st.subheader("Dividend discount & residual income (cost of equity based)")
    st.caption(f"Cost of equity {pct(ke)} = {pct(rf)} + β {beta:.2f} × {pct(erp)}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Dividend discount**")
        dps = c1.number_input("Dividend / share (₹)", value=float(F["dps"]) if finite(F["dps"]) else 0.0, step=0.5)
        g_div = c1.slider("Dividend growth — stage 1 %", 0.0, 30.0, round(default_g1 * 100 * 2) / 2, 0.5) / 100
        gor = v.gordon(dps, ke, g_term)
        ddm2 = v.ddm_two_stage(dps, g_div, ke, g_term, years, fade)
        c1.metric("Gordon (single-stage, g∞)", inr(gor), pct(up(gor, price), sign=True) if gor else None)
        c1.metric("Two-stage DDM", inr(ddm2), pct(up(ddm2, price), sign=True) if ddm2 else None)
        if ddm2:
            lo_d = v.ddm_two_stage(dps, g_div, ke + 0.01, g_term, years, fade)
            hi_d = v.ddm_two_stage(dps, g_div, ke - 0.01, g_term, years, fade) if ke - 0.01 > g_term else ddm2
            results["DDM (2-stage)"] = {"base": ddm2, "low": lo_d, "high": hi_d, "note": f"DPS ₹{dps:.2f}, g1 {pct(g_div)}"}
    with c2:
        st.markdown("**Residual income (Edwards-Bell-Ohlson)**")
        roe_in = c2.number_input("Current ROE %", value=round(F["roe"] * 100, 2) if finite(F["roe"]) else 15.0, step=0.5) / 100
        payout_in = c2.number_input("Payout ratio %", value=round(F["payout"] * 100, 1) if finite(F["payout"]) else 30.0, step=5.0) / 100
        spread = c2.slider("Terminal ROE spread over Ke %", 0.0, 10.0, float(aval("spread", AUTO["ri_spread"] * 100)), 0.5, **kw("spread"),
                           help="0 = competitive advantage fully fades. >0 = a durable moat earning this spread forever. "
                                f"Auto: {AN['ri_spread']}") / 100
        bvps = F["bvps"]
        ri = v.residual_income(bvps, roe_in, ke, payout_in, years, fade, spread, g_term) if finite(bvps) else None
        jpb = v.justified_pb(roe_in, ke, g_term)
        jpb_v = bvps * jpb if jpb and finite(bvps) and jpb > 0 else None
        c2.metric("Residual income value", inr(ri["per_share"]) if ri else "—", pct(up(ri["per_share"], price), sign=True) if ri else None)
        c2.metric("Justified P/B × BVPS", inr(jpb_v), f"P/B {jpb:.2f}x" if jpb else None)
        if ri:
            lo_r = v.residual_income(bvps, roe_in, ke + 0.01, payout_in, years, fade, spread, g_term)["per_share"]
            hi_r = v.residual_income(bvps, roe_in, ke - 0.01, payout_in, years, fade, spread, g_term)["per_share"]
            results["Residual income"] = {"base": ri["per_share"], "low": lo_r, "high": hi_r, "note": f"ROE {pct(roe_in)} → Ke+{pct(spread)}"}
            with st.expander("Residual income schedule"):
                t = ri["table"].copy()
                for col in ["ROE"]:
                    t[col] = t[col].map(pct)
                st.dataframe(t.round(2), hide_index=True, width="stretch")
        if jpb_v:
            results["Justified P/B"] = {"base": jpb_v, "low": jpb_v * 0.9, "high": jpb_v * 1.1, "note": f"(ROE−g)/(Ke−g) = {jpb:.2f}x"}

# ───────────────────────────── Multiples ─────────────────────────────
with tabs[5]:
    st.subheader("Relative valuation")
    eps = F["eps_ttm"]
    hist = yfd.get("history")
    band_years = st.slider("P/E history window (years)", 2, 15, 5)
    band = v.pe_band(hist["Close"], eps_annual, band_years) if hist is not None and eps_annual is not None else None
    if band and finite(eps) and eps > 0:
        fair_pe = eps * band["mean"]
        a, b, c = st.columns(3)
        a.metric("Current P/E", f"{band['current']:.1f}x", f"Z {band['z']:+.2f}", delta_color="inverse")
        b.metric(f"{band_years}-yr mean P/E", f"{band['mean']:.1f}x", f"σ {band['std']:.1f}")
        c.metric("Fair value at mean P/E", inr(fair_pe), pct(up(fair_pe, price), sign=True))
        sdf = band["series"]
        sdf = sdf[sdf.index >= band["window_start"] - pd.Timedelta(days=365)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sdf.index, y=sdf["eps"] * (band["mean"] + band["std"]), line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=sdf.index, y=sdf["eps"] * max(band["mean"] - band["std"], 0), fill="tonexty",
                                 fillcolor="rgba(120,144,156,0.18)", line=dict(width=0), name="±1σ P/E band"))
        fig.add_trace(go.Scatter(x=sdf.index, y=sdf["eps"] * band["mean"], name="EPS × mean P/E", line=dict(color="#78909C", dash="dash")))
        fig.add_trace(go.Scatter(x=sdf.index, y=sdf["price"], name="Price", line=dict(color="#FFB300", width=2)))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")
        results["P/E band (own history)"] = {"base": fair_pe, "low": eps * max(band["mean"] - band["std"], 0),
                                             "high": eps * (band["mean"] + band["std"]), "note": f"{band_years}y mean {band['mean']:.1f}x"}
    else:
        st.info("P/E band needs price history and a positive EPS history.")

    st.markdown("**Peers (Screener.in)**")
    peers = scr.get("peers")
    peer_pe_med = None
    if peers is not None and not peers.empty:
        pcols = {c: c for c in peers.columns}
        pe_col = next((c for c in peers.columns if c.replace(" ", "").upper().startswith("P/E")), None)
        sno = peers.columns[0]
        body = peers[pd.to_numeric(peers[sno], errors="coerce").notna()].copy()
        if pe_col:
            peer_pe_med = v.median_of(body[pe_col].map(ds.num))
        st.dataframe(body, hide_index=True, width="stretch", height=min(400, 38 * (len(body) + 1)))
        if peer_pe_med and finite(eps) and eps > 0:
            fv = eps * peer_pe_med
            st.metric("Fair value at peer median P/E", inr(fv), f"median {peer_pe_med:.1f}x · {pct(up(fv, price), sign=True)}")
            results["Peer P/E (Screener)"] = {"base": fv, "low": fv * 0.85, "high": fv * 1.15, "note": f"median {peer_pe_med:.1f}x"}
    else:
        st.caption(f"Screener peer table unavailable — {scr.get('peers_error') or scr.get('error') or 'unknown reason'}.")

    sector_pe = F.get("sector_pe")
    if finite(sector_pe) and sector_pe > 0 and finite(eps) and eps > 0:
        fv = eps * sector_pe
        st.metric("Fair value at sector P/E (Dhan)", inr(fv), f"sector {sector_pe:.1f}x · {pct(up(fv, price), sign=True)}")
        results["Sector P/E (Dhan)"] = {"base": fv, "low": fv * 0.85, "high": fv * 1.15, "note": f"sector {sector_pe:.1f}x"}

    st.markdown("**Custom peer set (yfinance multiples)**")
    peer_txt = st.text_input("Peer NSE symbols, comma-separated", value="", placeholder="e.g. HINDUNILVR, NESTLEIND, BRITANNIA", key="peer_txt")
    if peer_txt.strip():
        rows = []
        for p in [x.strip().upper() for x in peer_txt.split(",") if x.strip()][:12]:
            inf = get_peer_info(p)
            rows.append({"Symbol": p, "P/E": inf.get("trailingPE"), "P/B": inf.get("priceToBook"),
                         "EV/EBITDA": inf.get("enterpriseToEbitda"), "ROE %": (inf.get("returnOnEquity") or float("nan")) * 100})
        pdf = pd.DataFrame(rows)
        st.dataframe(pdf.round(2), hide_index=True, width="stretch")
        med_pe, med_pb, med_ev = v.median_of(pdf["P/E"]), v.median_of(pdf["P/B"]), v.median_of(pdf["EV/EBITDA"])
        cc = st.columns(3)
        if med_pe and finite(eps) and eps > 0:
            fv = eps * med_pe
            cc[0].metric("At peer P/E", inr(fv), f"{med_pe:.1f}x")
            results["Custom peers P/E"] = {"base": fv, "low": fv * 0.85, "high": fv * 1.15, "note": f"{med_pe:.1f}x"}
        if med_pb and finite(F["bvps"]):
            fv = F["bvps"] * med_pb
            cc[1].metric("At peer P/B", inr(fv), f"{med_pb:.2f}x")
            results["Custom peers P/B"] = {"base": fv, "low": fv * 0.85, "high": fv * 1.15, "note": f"{med_pb:.2f}x"}
        if med_ev and finite(F["ebitda"]) and finite(shares) and F["ebitda"] > 0 and not is_fin:
            fv = (F["ebitda"] * med_ev - net_debt) / shares
            cc[2].metric("At peer EV/EBITDA", inr(fv), f"{med_ev:.1f}x")
            if fv > 0:
                results["Custom peers EV/EBITDA"] = {"base": fv, "low": fv * 0.85, "high": fv * 1.15, "note": f"{med_ev:.1f}x"}

    if finite(F["target_mean"]):
        st.markdown("**Analyst consensus (yfinance)** — sentiment, not intrinsic value; excluded from the composite by default.")
        st.metric("Mean 12-m target", inr(F["target_mean"]),
                  f"{pct(up(F['target_mean'], price), sign=True)} · {int(F['analysts']) if finite(F['analysts']) else '?'} analysts")
        results["Analyst target"] = {"base": F["target_mean"], "low": F["target_low"] if finite(F["target_low"]) else F["target_mean"],
                                     "high": F["target_high"] if finite(F["target_high"]) else F["target_mean"], "note": "consensus"}

# ───────────────────────────── Graham & EPV ─────────────────────────────
with tabs[6]:
    st.subheader("Formula methods")
    eps, bvps = F["eps_ttm"], F["bvps"]
    c1, c2 = st.columns(2)
    gg_g = c1.slider("Growth for Graham formula %", 0.0, 20.0, min(round(default_g1 * 100 * 2) / 2, 20.0), 0.5) / 100
    y_pct = c1.number_input("Bond yield Y % (10Y G-Sec)", value=round(rf * 100, 2), step=0.05)
    gn = v.graham_number(eps, bvps)
    gg = v.graham_growth(eps, gg_g, y_pct)
    c1.metric("Graham Number √(22.5·EPS·BVPS)", inr(gn), pct(up(gn, price), sign=True) if gn else None)
    c1.metric("Graham growth EPS·(8.5+2g)·4.4/Y", inr(gg), pct(up(gg, price), sign=True) if gg else None)
    c1.caption("Graham Number assumes P/E ≤ 15 and P/B ≤ 1.5 — very conservative for Indian quality compounders.")
    epv_rate = c2.number_input("EPV capitalisation rate %", value=round(wacc_ * 100, 2), step=0.25) / 100
    ep = v.epv(F["ebit"], tax, epv_rate, net_debt, shares) if not is_fin else None
    c2.metric("Earnings Power Value", inr(ep), pct(up(ep, price), sign=True) if ep else None)
    c2.caption("EPV = EBIT·(1−t)/cost of capital − net debt: the business worth with zero growth. "
               "Price − EPV ≈ what the market pays for growth.")
    if ep and dcf_key in results:
        c2.metric("Growth premium in DCF over EPV", inr(results[dcf_key]["base"] - ep))
    if gn:
        results["Graham Number"] = {"base": gn, "low": gn, "high": gn, "note": "15×P/E · 1.5×P/B"}
    if gg:
        results["Graham growth"] = {"base": gg, "low": v.graham_growth(eps, max(gg_g - 0.02, 0), y_pct),
                                    "high": v.graham_growth(eps, gg_g + 0.02, y_pct), "note": f"g {pct(gg_g)}, Y {y_pct:.2f}%"}
    if ep and ep > 0:
        results["EPV"] = {"base": ep, "low": v.epv(F["ebit"], tax, epv_rate + 0.01, net_debt, shares),
                          "high": v.epv(F["ebit"], tax, max(epv_rate - 0.01, 0.01), net_debt, shares), "note": f"r {pct(epv_rate)}"}

# ───────────────────────────── SOTP ─────────────────────────────
with tabs[7]:
    st.subheader("Sum of the parts")
    st.caption("Segment data isn't available from these sources — fill segments from the annual report / investor deck. "
               "Metric is whatever you're applying the multiple to (EBITDA, PAT, book value…) in ₹ crore.")
    seg = st.data_editor(pd.DataFrame({"Segment": ["", "", ""], "Metric (₹ Cr)": [None, None, None], "Multiple": [None, None, None]}),
                         key=f"seg_{symbol}", num_rows="dynamic", hide_index=True, width="stretch",
                         column_config={"Metric (₹ Cr)": st.column_config.NumberColumn(format="%.1f"),
                                        "Multiple": st.column_config.NumberColumn(format="%.1f")})
    hold = st.data_editor(pd.DataFrame({"Listed holding": ["", ""], "Value (₹ Cr)": [None, None]}),
                          key=f"hold_{symbol}", num_rows="dynamic", hide_index=True, width="stretch",
                          column_config={"Value (₹ Cr)": st.column_config.NumberColumn("Attributable stake value (₹ Cr)", format="%.1f")})
    hd = st.slider("Holding-company discount %", 0, 50, 20) / 100
    for col in ["Metric (₹ Cr)", "Multiple"]:
        seg[col] = pd.to_numeric(seg[col], errors="coerce")
    hold["Value (₹ Cr)"] = pd.to_numeric(hold["Value (₹ Cr)"], errors="coerce")
    so = v.sotp(seg, hold, hd, net_debt, shares) if finite(shares) else None
    if so:
        a, b, c = st.columns(3)
        a.metric("Segments", crore(so["segments"]))
        b.metric("Holdings after discount", crore(so["holdings"]))
        c.metric("SOTP / share", inr(so["per_share"]), pct(up(so["per_share"], price), sign=True))
        results["SOTP"] = {"base": so["per_share"], "low": so["per_share"] * 0.9, "high": so["per_share"] * 1.1, "note": f"holdco discount {hd:.0%}"}

# ───────────────────────────── Company (Dhan) ─────────────────────────────
with tabs[8]:
    st.subheader("Company information")
    if ext.get("summary"):
        st.write(ext["summary"])
    if dh.get("ok"):
        dx = dh.get("extract") or {}
        shown = {k: v_ for k, v_ in dx.items() if not k.endswith("__key")}
        if shown:
            st.markdown("**Fields recognised from Dhan**")
            st.dataframe(pd.DataFrame({"Field": list(shown), "Value": [str(x) for x in shown.values()],
                                       "Dhan key": [dx.get(f"{k}__key") for k in shown]}), hide_index=True, width="stretch")
        data = dh.get("data") or {}
        sections = data.items() if isinstance(data, dict) else [("data", data)]
        for sec_name, sec_val in sections:
            with st.expander(f"Dhan section: {sec_name}", expanded=False):
                flat = ds.flatten(sec_val)
                st.dataframe(pd.DataFrame({"Key": list(flat), "Value": [str(x) for x in flat.values()]}),
                             hide_index=True, width="stretch")
        with st.expander("Raw Dhan response (JSON)"):
            st.json(dh.get("raw") or data, expanded=False)
    else:
        st.info(f"Dhan company info unavailable: {dh.get('error')}")
    shp = scr.get("shp")
    if shp is not None:
        st.markdown("**Shareholding pattern (Screener.in)**")
        st.dataframe(shp.iloc[:, -8:], width="stretch")
    if scr.get("ratios_raw"):
        st.markdown("**Screener headline ratios**")
        st.dataframe(pd.DataFrame({"Ratio": list(scr["ratios_raw"]), "Value": list(scr["ratios_raw"].values())}),
                     hide_index=True, width="stretch")

# ───────────────────────────── Summary ─────────────────────────────
with tabs[0]:
    if not results:
        st.warning("No method produced a value — check the Inputs tab for missing data.")
    else:
        avoid = {"Analyst target"} | ({"DCF (FCFF)", "DCF (FCFE)", "EPV", "Custom peers EV/EBITDA", "DDM (2-stage)"} if is_fin else set())
        default_inc = [k for k in results if k not in avoid]
        include = st.multiselect("Methods in the composite (median)", list(results), default=default_inc)
        comp = v.composite(results, include)
        if comp:
            fv = comp["fair_value"]
            buy = fv * (1 - mos)
            label, colour = v.verdict(price, fv, mos)
            a, b, c, d = st.columns(4)
            a.metric("Price", inr(price))
            b.metric("Composite fair value", inr(fv), pct(up(fv, price), sign=True))
            c.metric(f"Buy below ({mos:.0%} MoS)", inr(buy), pct(up(buy, price), sign=True))
            d.metric("Confidence", comp["confidence"], f"{comp['n']} methods · CV {pct(comp['cv'], 0)}", delta_color="off")
            st.markdown(f"<div style='padding:10px 14px;border-radius:8px;border-left:6px solid {colour};"
                        f"background:rgba(127,127,127,0.08);font-weight:600'>{label}</div>", unsafe_allow_html=True)
            st.caption(f"Interquartile range of included methods: {inr(comp['low'])} – {inr(comp['high'])}")

        names = list(results)[::-1]
        fig = go.Figure()
        for n_ in names:
            r_ = results[n_]
            lo_, hi_ = r_.get("low"), r_.get("high")
            lo_ = lo_ if finite(lo_) else r_["base"]
            hi_ = hi_ if finite(hi_) else r_["base"]
            lo_, hi_ = min(lo_, hi_), max(lo_, hi_)
            inc = n_ in include
            fig.add_trace(go.Bar(y=[n_], x=[max(hi_ - lo_, price * 0.005)], base=[lo_], orientation="h",
                                 marker_color="#26A69A" if inc else "#B0BEC5", opacity=0.85 if inc else 0.5,
                                 showlegend=False, hovertemplate=f"{n_}: {inr(lo_, 0)} – {inr(hi_, 0)}<extra></extra>"))
            fig.add_trace(go.Scatter(y=[n_], x=[r_["base"]], mode="markers", marker=dict(color="#263238", size=9, symbol="diamond"),
                                     showlegend=False, hovertemplate=f"{n_} base {inr(r_['base'], 0)}<extra></extra>"))
        fig.add_vline(x=price, line=dict(color="#FFB300", width=2), annotation_text="Price", annotation_position="top")
        if comp:
            fig.add_vline(x=comp["fair_value"], line=dict(color="#1E88E5", width=2, dash="dash"), annotation_text="Fair", annotation_position="bottom")
            fig.add_vline(x=comp["fair_value"] * (1 - mos), line=dict(color="#43A047", width=1, dash="dot"), annotation_text="Buy", annotation_position="bottom")
        fig.update_layout(title="Football field — value range by method (◆ = base case)", barmode="overlay",
                          height=max(320, 46 * len(names) + 90), margin=dict(l=10, r=10, t=50, b=10), xaxis_title="₹ per share")
        st.plotly_chart(fig, width="stretch")

        tbl = pd.DataFrame([{"Method": k, "Fair value": r_["base"], "Low": r_.get("low"), "High": r_.get("high"),
                             "Upside %": up(r_["base"], price) * 100, "In composite": "✓" if k in include else "",
                             "Assumptions": r_.get("note", "")} for k, r_ in results.items()])
        st.dataframe(tbl, hide_index=True, width="stretch",
                     column_config={c_: st.column_config.NumberColumn(format="%.1f") for c_ in ["Fair value", "Low", "High", "Upside %"]})
        st.caption(f"Ke {pct(ke)} · WACC {pct(wacc_)} · g∞ {pct(g_term)} · {years} explicit years · "
                   f"generated {datetime.now():%d %b %Y %H:%M}. Educational tool, not investment advice.")
