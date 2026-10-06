import datetime as dt

import pandas as pd
import pytest

import ibkr_flex
from ibkr_flex import AccountHistory, FlexError

SAMPLE = """<FlexQueryResponse queryName="perf" type="AF">
<FlexStatements count="1">
<FlexStatement accountId="U1234567" fromDate="20260201" toDate="20260210">
<EquitySummaryInBase>
<EquitySummaryByReportDateInBase accountId="U1234567" currency="SGD" reportDate="20260203" total="1000.50" />
<EquitySummaryByReportDateInBase accountId="U1234567" currency="SGD" reportDate="20260204" total="1010.25" />
</EquitySummaryInBase>
<CashTransactions>
<CashTransaction type="Deposits &amp; Withdrawals" currency="SGD" fxRateToBase="1" amount="1000" dateTime="20260203;101500" reportDate="20260203" levelOfDetail="DETAIL" />
<CashTransaction type="Deposits &amp; Withdrawals" currency="USD" fxRateToBase="1.30" amount="-100" dateTime="20260204;090000" reportDate="20260204" levelOfDetail="DETAIL" />
<CashTransaction type="Dividends" currency="USD" fxRateToBase="1.30" amount="5" dateTime="20260204;090000" reportDate="20260204" levelOfDetail="DETAIL" />
<CashTransaction type="Deposits &amp; Withdrawals" currency="SGD" fxRateToBase="1" amount="900" reportDate="20260204" levelOfDetail="SUMMARY" />
</CashTransactions>
</FlexStatement>
</FlexStatements>
</FlexQueryResponse>"""


def test_parse_nav_and_flows():
    nav, flows, currency = ibkr_flex.parse_flex_xml(SAMPLE)
    assert currency == "SGD"
    assert nav.loc["2026-02-03"] == 1000.50 and nav.loc["2026-02-04"] == 1010.25
    assert flows.loc["2026-02-03"] == 1000.0
    assert flows.loc["2026-02-04"] == pytest.approx(-130.0)   # USD amount x FX, dividends and summary rows ignored


def test_parse_sums_accounts_per_date():
    xml = """<FlexQueryResponse><FlexStatements>
    <FlexStatement><EquitySummaryInBase>
      <EquitySummaryByReportDateInBase currency="USD" reportDate="2026-02-03" total="100" />
    </EquitySummaryInBase></FlexStatement>
    <FlexStatement><EquitySummaryInBase>
      <EquitySummaryByReportDateInBase currency="USD" reportDate="2026-02-03" total="250" />
    </EquitySummaryInBase></FlexStatement>
    </FlexStatements></FlexQueryResponse>"""
    nav, flows, _ = ibkr_flex.parse_flex_xml(xml)
    assert nav.loc["2026-02-03"] == 350 and flows.empty


def test_parse_empty_window():
    nav, flows, currency = ibkr_flex.parse_flex_xml("<FlexQueryResponse><FlexStatements/></FlexQueryResponse>")
    assert nav.empty and flows.empty and currency == "USD"


def test_summary_only_cash_rows_is_an_actionable_error():
    xml = """<FlexQueryResponse><CashTransactions>
    <CashTransaction type="Deposits &amp; Withdrawals" amount="5" reportDate="20260203" levelOfDetail="SUMMARY" />
    </CashTransactions></FlexQueryResponse>"""
    with pytest.raises(FlexError, match="Detail"):
        ibkr_flex.parse_flex_xml(xml)


def test_missing_total_field_is_an_actionable_error():
    xml = """<FlexQueryResponse><EquitySummaryInBase>
    <EquitySummaryByReportDateInBase reportDate="20260203" cash="5" />
    </EquitySummaryInBase></FlexQueryResponse>"""
    with pytest.raises(FlexError, match="Total"):
        ibkr_flex.parse_flex_xml(xml)


def test_unreadable_date_format_is_an_actionable_error():
    xml = """<FlexQueryResponse><EquitySummaryInBase>
    <EquitySummaryByReportDateInBase reportDate="02/03/2026" total="5" />
    </EquitySummaryInBase></FlexQueryResponse>"""
    with pytest.raises(FlexError, match="yyyyMMdd"):
        ibkr_flex.parse_flex_xml(xml)


# --- request flow -----------------------------------------------------------

def fake_get(responses, calls):
    def _get(endpoint, **params):
        calls.append((endpoint, params))
        return responses.pop(0)
    return _get


SEND_OK = "<FlexStatementResponse><Status>Success</Status><ReferenceCode>REF123</ReferenceCode></FlexStatementResponse>"
IN_PROGRESS = ("<FlexStatementResponse><Status>Warn</Status><ErrorCode>1019</ErrorCode>"
               "<ErrorMessage>Statement generation in progress. Please try again shortly.</ErrorMessage></FlexStatementResponse>")
BAD_TOKEN = ("<FlexStatementResponse><Status>Fail</Status><ErrorCode>1015</ErrorCode>"
             "<ErrorMessage>Token is invalid.</ErrorMessage></FlexStatementResponse>")


def test_fetch_statement_polls_until_ready(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(ibkr_flex, "_get", fake_get([SEND_OK, IN_PROGRESS, IN_PROGRESS, SAMPLE], calls))
    xml = ibkr_flex.fetch_statement("tok", "42", dt.date(2026, 2, 1), dt.date(2026, 2, 10), sleep=sleeps.append)
    assert xml == SAMPLE
    assert [c[0] for c in calls] == ["SendRequest", "GetStatement", "GetStatement", "GetStatement"]
    assert calls[0][1]["fd"] == "20260201" and calls[0][1]["td"] == "20260210" and calls[0][1]["v"] == 3
    assert calls[1][1]["q"] == "REF123"
    assert len(sleeps) == 2


def test_fetch_statement_raises_on_hard_error(monkeypatch):
    monkeypatch.setattr(ibkr_flex, "_get", fake_get([BAD_TOKEN], []))
    with pytest.raises(FlexError, match="1015.*Token is invalid"):
        ibkr_flex.fetch_statement("tok", "42", dt.date(2026, 2, 1), dt.date(2026, 2, 10), sleep=lambda s: None)


def test_http_errors_never_leak_the_token(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("Max retries exceeded with url: /SendRequest?t=SECRETTOKEN&q=1")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(FlexError) as e:
        ibkr_flex._get("SendRequest", t="SECRETTOKEN", q="1", v=3)
    assert "SECRETTOKEN" not in str(e.value)


# --- history windows --------------------------------------------------------

def xml_for(window_start, window_end, first_funded):
    """Fake Flex statement: NAV of 1000 on each Friday from `first_funded` on."""
    days = pd.date_range(window_start, window_end, freq="W-FRI")
    rows = "".join(
        f'<EquitySummaryByReportDateInBase currency="USD" reportDate="{d:%Y%m%d}" total="{1000 if d >= first_funded else 0}" />'
        for d in days
    )
    return f"<FlexQueryResponse><EquitySummaryInBase>{rows}</EquitySummaryInBase></FlexQueryResponse>"


def test_history_walks_back_until_inception_and_respects_365_day_cap():
    first_funded = pd.Timestamp("2025-03-14")
    seen = []

    def fake_fetch(token, qid, start, end):
        seen.append((start, end))
        assert (end - start).days + 1 <= 365
        return xml_for(start, end, first_funded)

    h = ibkr_flex.fetch_history("t", "q", end=dt.date(2026, 10, 6), fetch=fake_fetch)
    assert len(seen) == 2                                  # window 2 contains inception: stop
    assert seen[1][1] == seen[0][0] - dt.timedelta(days=1)  # windows are contiguous
    funded = h.nav[h.nav > 0]
    assert funded.index.min() == first_funded
    assert h.nav.index.is_monotonic_increasing and h.nav.index.is_unique


def test_history_with_explicit_start_does_not_probe_further_back():
    seen = []

    def fake_fetch(token, qid, start, end):
        seen.append((start, end))
        return xml_for(start, end, pd.Timestamp("2026-01-02"))

    ibkr_flex.fetch_history("t", "q", start=dt.date(2026, 1, 1), end=dt.date(2026, 10, 6), fetch=fake_fetch)
    assert seen == [(dt.date(2026, 1, 1), dt.date(2026, 10, 6))]


def test_history_raises_when_nothing_is_funded():
    def fake_fetch(token, qid, start, end):
        return "<FlexQueryResponse/>"

    with pytest.raises(FlexError, match="No NAV rows"):
        ibkr_flex.fetch_history("t", "q", end=dt.date(2026, 10, 6), fetch=fake_fetch)


def test_cache_round_trip_and_expiry(tmp_path):
    nav, flows, cur = ibkr_flex.parse_flex_xml(SAMPLE)
    ibkr_flex.save_cache(AccountHistory(nav, flows, cur), tmp_path)

    loaded = ibkr_flex.load_cache(tmp_path)
    pd.testing.assert_series_equal(loaded.nav, nav.rename("nav"), check_freq=False, check_names=False)
    pd.testing.assert_series_equal(loaded.flows, flows.rename("flows"), check_freq=False, check_names=False)
    assert loaded.currency == "SGD"

    assert ibkr_flex.load_cache(tmp_path, max_age_hours=0) is None
    assert ibkr_flex.load_cache(tmp_path / "missing") is None
