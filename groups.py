"""Valuation groups — the same 8 groups as the Pine indicator "Shantanu's Valuation Matrix with Z-Score".

The group decides which methods enter the composite by default. Resolution order:
  1. your per-stock choice in the sidebar   2. the hand-checked Nifty 500 list (nifty500_valuation_groups.csv,
  NSE constituents of 28 Oct 2026)   3. yfinance industry   4. yfinance sector   5. General.
yfinance (Yahoo/Morningstar) industry names differ from TradingView's, so steps 3–4 use their own keyword map;
check the result on a few stocks outside the Nifty 500.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LIST_DATE = "28 Oct 2026"
CSV_PATH = Path(__file__).parent / "nifty500_valuation_groups.csv"

GROUPS = {   # code: (label, colour)
    "GEN": ("General / compounder", "#1E88E5"),
    "LEN": ("Lender (bank / NBFC)", "#00897B"),
    "INS": ("Insurer", "#3949AB"),
    "CAP": ("Capital-heavy", "#6A1B9A"),
    "CYC": ("Cyclical (mid-cycle)", "#EF6C00"),
    "YLD": ("Yield (REIT / utility)", "#43A047"),
    "HLD": ("Holding co / conglomerate", "#6D4C41"),
    "PLT": ("Platform / new-age", "#D81B60"),
}
LABEL_TO_CODE = {lab: code for code, (lab, _) in GROUPS.items()}

# Method names exactly as app.py stores them in `results`.
DCF = ["DCF (FCFF)", "DCF (FCFE)"]
METHODS = {
    "GEN": ["Graham Number", "Graham growth", *DCF, "DDM (2-stage)", "EPV", "P/E band (own history)"],
    "LEN": ["Graham Number", "P/E band (own history)", "P/B band (own history)", "Justified P/B", "Residual income",
            "DDM (2-stage)"],
    "INS": ["Graham Number", "P/E band (own history)", "P/B band (own history)", "Justified P/B", "Residual income",
            "DDM (2-stage)"],
    "CAP": ["Graham Number", *DCF, "DDM (2-stage)", "EPV", "P/E band (own history)", "EV/EBITDA band (own history)"],
    "CYC": ["Graham Number", "DDM (2-stage)", "EPV", "P/B band (own history)", "EV/EBITDA band (own history)"],
    "YLD": ["DDM (2-stage)", "P/E band (own history)", "P/B band (own history)", "Justified P/B"],
    "HLD": ["Graham Number", "P/E band (own history)", "P/B band (own history)", "DDM (2-stage)", "SOTP"],
    "PLT": ["EV/Sales band (own history)", *DCF, "P/E band (own history)"],
}
# Cash-flow / EV methods that don't fit financial balance sheets (switched off, as in the Pine script)
FIN_GROUPS = {"LEN", "INS", "HLD"}
# Groups where the DDM enters whenever a dividend exists (not only for steady payers)
DDM_ALWAYS = {"YLD"}
# Groups whose EV/EBITDA band uses mid-cycle (average of the last N fiscal years) EBITDA
MID_CYCLE = {"CYC"}

NOTES = {
    "INS": "Embedded value / VNB are not available from these sources — book-value methods and residual income are used.",
    "HLD": "Fill the 🧩 SOTP tab (segments + listed stakes) — sum-of-the-parts is the main method for holding companies.",
    "CYC": "EV/EBITDA uses mid-cycle EBITDA so peak-year profits don't make the stock look cheap.",
    "PLT": "Profits are not yet steady — EV/Sales is the main yardstick; P/E-based values may be missing or extreme.",
    "YLD": "Dividend yield versus the 10Y G-Sec matters most; the DDM is used whenever a dividend is paid.",
}


def load_list(path: Path = CSV_PATH) -> dict[str, tuple[str, str]]:
    """NSE symbol → (group code, reason)."""
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return {}
    out = {}
    for sym, grp, why in zip(df["NSE symbol"], df["Valuation group"], df["Reason"]):
        code = LABEL_TO_CODE.get(str(grp).strip())
        if code:
            out[str(sym).strip().upper()] = (code, str(why))
    return out


# yfinance industry keywords (lower case, matched with `in`), checked in this order.
YF_INDUSTRY = [
    ("GEN", ["insurance brokers"]),
    ("INS", ["insurance"]),
    ("LEN", ["banks", "credit services", "mortgage finance"]),
    ("HLD", ["conglomerates"]),
    ("YLD", ["reit", "utilities - regulated electric"]),
    ("CYC", ["steel", "aluminum", "copper", "industrial metals", "coal", "oil & gas integrated", "oil & gas e&p",
             "oil & gas refining", "real estate - development", "real estate—development", "marine shipping",
             "paper & paper", "agricultural inputs"]),
    ("CAP", ["building materials", "medical care facilities", "telecom services", "airlines", "utilities",
             "lodging", "airports", "railroads", "oil & gas midstream"]),
]
YF_SECTOR = [("CYC", ["energy", "real estate"]), ("CAP", ["utilities"])]


def from_yf(industry: str | None, sector: str | None) -> tuple[str | None, str]:
    ind, sec = (industry or "").lower(), (sector or "").lower()
    for code, keys in YF_INDUSTRY:
        if any(k in ind for k in keys):
            return code, "yfinance industry"
    for code, keys in YF_SECTOR:
        if any(k in sec for k in keys):
            return code, "yfinance sector"
    return None, ""


def resolve(symbol: str, industry: str | None, sector: str | None, choice: str | None,
            table: dict[str, tuple[str, str]]) -> dict:
    """Returns {"code", "label", "colour", "basis", "reason"}. `choice` = a group code or None for Auto."""
    sym = (symbol or "").upper().strip()
    if choice in GROUPS:
        code, basis, why = choice, "your choice", "set in the sidebar"
    elif sym in table:
        code, why = table[sym]
        basis = f"Nifty 500 list ({LIST_DATE})"
    else:
        code, basis = from_yf(industry, sector)
        why = f"{industry or '—'} / {sector or '—'}"
        if code is None:
            code, basis, why = "GEN", "default", "no match in the list or yfinance classification"
    lab, col = GROUPS[code]
    return {"code": code, "label": lab, "colour": col, "basis": basis, "reason": why}
