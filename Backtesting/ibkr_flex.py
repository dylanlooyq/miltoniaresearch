"""Account NAV history and deposits/withdrawals from IBKR's Flex Web Service.

Flex is read-only and needs no running TWS/Gateway. One-time setup (create the
Flex Query + token) is described in README.md.
"""
import datetime as dt
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"
MAX_WINDOW_DAYS = 365          # Flex Web Service caps each request at 365 days
CASH_FLOW_TYPE = "Deposits & Withdrawals"
POLL_SECONDS = 5
RATE_LIMIT_SECONDS = 15
MAX_WAIT_SECONDS = 180
RETRY_CODES = {"1001", "1009", "1018", "1019", "1021"}


class FlexError(RuntimeError):
    pass


@dataclass
class AccountHistory:
    nav: pd.Series      # total account value (base currency) by report date
    flows: pd.Series    # net deposits (+) / withdrawals (-) (base currency) by date
    currency: str


# -------------------
# Credentials
# -------------------
def credentials():
    load_dotenv()
    token = os.getenv("IBKR_FLEX_TOKEN")
    query_id = os.getenv("IBKR_FLEX_QUERY_ID")
    if not token or not query_id:
        raise FlexError(
            "Set IBKR_FLEX_TOKEN and IBKR_FLEX_QUERY_ID in a .env file "
            "(copy .env.example; setup steps are in README.md)."
        )
    return token, query_id


# -------------------
# Flex Web Service requests
# -------------------
def _get(endpoint, **params):
    # requests puts the full URL (including the token) in its exception text,
    # so never let those propagate.
    try:
        r = requests.get(
            f"{BASE_URL}/{endpoint}",
            params=params,
            headers={"User-Agent": "miltonia-research"},
            timeout=60,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        detail = f"HTTP {status}" if status else type(e).__name__
        raise FlexError(f"IBKR {endpoint} request failed ({detail})") from None
    return r.text


def _parse(text):
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        raise FlexError("IBKR returned a response that is not XML") from None


def _call(endpoint, deadline, sleep, **params):
    """Call a Flex endpoint, retrying while IBKR says to try again shortly."""
    while True:
        text = _get(endpoint, **params)
        root = _parse(text)
        if root.tag == "FlexQueryResponse" or root.findtext("Status") == "Success":
            return root, text
        code = root.findtext("ErrorCode") or "?"
        message = root.findtext("ErrorMessage") or "no message"
        retryable = code in RETRY_CODES or "try again" in message.lower()
        if not retryable or time.monotonic() > deadline:
            raise FlexError(f"IBKR error {code}: {message}")
        sleep(RATE_LIMIT_SECONDS if code == "1018" else POLL_SECONDS)


def fetch_statement(token, query_id, start, end, sleep=time.sleep):
    """Run the Flex Query for [start, end] (<= 365 days) and return the XML text."""
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    sent, _ = _call(
        "SendRequest", deadline, sleep,
        t=token, q=query_id, v=3,
        fd=start.strftime("%Y%m%d"), td=end.strftime("%Y%m%d"),
    )
    reference = sent.findtext("ReferenceCode")
    if not reference:
        raise FlexError("IBKR accepted the request but returned no reference code")
    _, text = _call("GetStatement", deadline, sleep, t=token, q=reference, v=3)
    return text


# -------------------
# Parsing
# -------------------
def _to_date(value):
    """Flex dates: 20260215, 2026-02-15, 20260215;093000, ... -> Timestamp."""
    digits = re.sub(r"\D", "", value or "")[:8]
    try:
        return pd.Timestamp(dt.datetime.strptime(digits, "%Y%m%d"))
    except ValueError:
        raise FlexError(
            f"Cannot read date {value!r} - set the Flex Query's Date Format to yyyyMMdd"
        ) from None


def _empty_series():
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def parse_flex_xml(xml_text):
    """Flex XML -> (nav, flows, currency); several accounts are summed per date."""
    root = _parse(xml_text)

    nav, currency = {}, None
    for e in root.iter("EquitySummaryByReportDateInBase"):
        total = e.get("total")
        if total is None:
            raise FlexError(
                "NAV rows have no 'total' - add the Total field to the "
                "'Net Asset Value (NAV) in Base' section of the Flex Query"
            )
        if total == "":
            continue
        day = _to_date(e.get("reportDate"))
        nav[day] = nav.get(day, 0.0) + float(total)
        currency = currency or e.get("currency")

    cash = [e for e in root.iter("CashTransaction") if e.get("type") == CASH_FLOW_TYPE]
    detail = [e for e in cash if e.get("levelOfDetail") in (None, "", "DETAIL")]
    if cash and not detail:
        raise FlexError(
            "Cash Transactions came back as summary rows only - tick 'Detail' in "
            "that section's options in the Flex Query"
        )
    flows = {}
    for e in detail:
        day = _to_date(e.get("reportDate") or e.get("dateTime") or e.get("settleDate"))
        amount = float(e.get("amount") or 0) * float(e.get("fxRateToBase") or 1)
        flows[day] = flows.get(day, 0.0) + amount

    nav = pd.Series(nav, dtype=float).sort_index() if nav else _empty_series()
    flows = pd.Series(flows, dtype=float).sort_index() if flows else _empty_series()
    return nav, flows, currency or "USD"


# -------------------
# Full history
# -------------------
def fetch_history(token, query_id, start=None, end=None, max_years=10, fetch=fetch_statement):
    """NAV + deposits from `start` (default: detected inception) to `end` (default: today).

    Walks backwards in <=365-day windows. With no `start`, it stops once a window
    shows the account's first funded day (or no data).
    """
    end = end or dt.date.today()
    earliest = start or end - dt.timedelta(days=365 * max_years)

    navs, flows, currency = [], [], None
    window_end = end
    while window_end >= earliest:
        window_start = max(window_end - dt.timedelta(days=MAX_WINDOW_DAYS - 1), earliest)
        try:
            xml_text = fetch(token, query_id, window_start, window_end)
        except FlexError as e:
            if not navs:
                raise
            print(f"Stopping history lookup before {window_end}: {e}")
            break
        n, f, cur = parse_flex_xml(xml_text)
        navs.append(n)
        flows.append(f)
        currency = currency or cur
        if start is None:
            funded = n[n > 0]
            if funded.empty or funded.index.min() > pd.Timestamp(window_start) + pd.Timedelta(days=7):
                break
        window_end = window_start - dt.timedelta(days=1)

    nav = pd.concat(navs).sort_index()
    nav = nav[~nav.index.duplicated(keep="first")]
    if not (nav > 0).any():
        raise FlexError(
            "No NAV rows returned - check the Flex Query includes "
            "'Net Asset Value (NAV) in Base' with 'Breakout by Day' ticked"
        )
    flow = pd.concat(flows).sort_index().groupby(level=0).sum()
    return AccountHistory(nav=nav, flows=flow, currency=currency or "USD")


# -------------------
# Cache (Flex is rate limited and slow; re-runs shouldn't need it)
# -------------------
def save_cache(history, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    history.nav.rename("nav").to_csv(cache_dir / "ibkr_nav.csv", index_label="date")
    history.flows.rename("flows").to_csv(cache_dir / "ibkr_flows.csv", index_label="date")
    meta = {"currency": history.currency, "fetched": dt.datetime.now().isoformat()}
    (cache_dir / "ibkr_meta.json").write_text(json.dumps(meta))


def load_cache(cache_dir, max_age_hours=12):
    """Cached AccountHistory, or None if missing/older than max_age_hours."""
    cache_dir = Path(cache_dir)
    try:
        meta = json.loads((cache_dir / "ibkr_meta.json").read_text())
        age = dt.datetime.now() - dt.datetime.fromisoformat(meta["fetched"])
        if age > dt.timedelta(hours=max_age_hours):
            return None
        nav = pd.read_csv(cache_dir / "ibkr_nav.csv", index_col="date")["nav"]
        flows = pd.read_csv(cache_dir / "ibkr_flows.csv", index_col="date")["flows"]
    except (OSError, ValueError, KeyError):
        return None
    nav.index = pd.to_datetime(nav.index)
    flows.index = pd.to_datetime(flows.index)
    return AccountHistory(nav=nav, flows=flows, currency=meta["currency"])
