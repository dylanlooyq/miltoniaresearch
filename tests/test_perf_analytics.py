import numpy as np
import pandas as pd
import pytest

from perf_analytics import (
    align_to_trading_days,
    benchmark_with_same_flows,
    build_comparison,
    performance_stats,
    summarize,
    twr_returns,
)


def days(n, start="2026-02-02"):
    return pd.bdate_range(start, periods=n)


def test_deposit_is_not_performance():
    # Day 3: +10% market move on 1,100, then 1,000 deposited at day end.
    d = days(3)
    nav = pd.Series([1000.0, 1100.0, 2210.0], index=d)
    flows = pd.Series([0.0, 0.0, 1000.0], index=d)
    r = twr_returns(nav, flows)
    assert r.iloc[0] == pytest.approx(0.10)
    assert r.iloc[1] == pytest.approx(0.10)


def test_withdrawal_is_not_a_loss():
    d = days(2)
    nav = pd.Series([2000.0, 1000.0], index=d)   # flat market, 1,000 withdrawn
    flows = pd.Series([0.0, -1000.0], index=d)
    assert twr_returns(nav, flows).iloc[0] == pytest.approx(0.0)


def test_zero_nav_day_does_not_produce_inf():
    d = days(4)
    nav = pd.Series([1000.0, 0.0, 0.0, 500.0], index=d)
    flows = pd.Series([0.0, -1000.0, 0.0, 500.0], index=d)
    r = twr_returns(nav, flows)
    assert np.isfinite(r).all()
    assert r.iloc[-1] == 0.0   # previous NAV was zero: no return defined


def test_portfolio_that_holds_the_benchmark_matches_it_exactly():
    rng = np.random.default_rng(1)
    d = days(60)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(d))), index=d)
    flows = pd.Series(0.0, index=d)
    flows.iloc[[10, 25, 40]] = [500.0, 1500.0, -700.0]

    nav = benchmark_with_same_flows(prices, flows, start_value=1000.0)
    r = twr_returns(nav, flows)
    np.testing.assert_allclose(r, prices.pct_change().iloc[1:], atol=1e-12)


def test_same_flows_buys_at_that_days_close():
    d = days(3)
    prices = pd.Series([100.0, 110.0, 121.0], index=d)
    flows = pd.Series([0.0, 110.0, 0.0], index=d)   # buys 1 unit on day 2
    v = benchmark_with_same_flows(prices, flows, start_value=1000.0)
    assert v.iloc[0] == pytest.approx(1000.0)
    assert v.iloc[2] == pytest.approx(10 * 121 + 1 * 121)


def test_weekend_deposit_rolls_to_next_trading_day():
    d = days(5)                                          # Mon 2 Feb .. Fri 6 Feb
    nav = pd.Series(1000.0, index=pd.date_range("2026-02-02", "2026-02-06"))
    flows = pd.Series([250.0], index=pd.DatetimeIndex(["2026-02-07"]))  # Saturday
    d = pd.bdate_range("2026-02-02", "2026-02-10")
    nav = pd.Series(1000.0, index=pd.date_range("2026-02-02", "2026-02-10"))
    _, flows_d = align_to_trading_days(nav, flows, d)
    assert flows_d.loc["2026-02-09"] == 250.0            # Monday
    assert flows_d.sum() == 250.0


def test_flow_after_last_trading_day_is_dropped():
    d = pd.bdate_range("2026-02-02", "2026-02-06")
    nav = pd.Series(1000.0, index=d)
    flows = pd.Series([99.0], index=pd.DatetimeIndex(["2026-02-08"]))
    _, flows_d = align_to_trading_days(nav, flows, d)
    assert flows_d.sum() == 0.0


def test_build_comparison_starts_at_first_funded_day():
    d = days(10)
    prices = pd.Series(np.linspace(100, 110, 10), index=d)
    nav = pd.Series([0, 0, 0, 1000, 1010, 1020, 1030, 1040, 1050, 1060.0], index=d)
    df = build_comparison(nav, pd.Series(dtype=float, index=pd.DatetimeIndex([])), prices)
    assert df.index[0] == d[3] and df.index[-1] == d[-1]
    assert df["portfolio_growth"].iloc[0] == 1.0 and df["benchmark_growth"].iloc[0] == 1.0
    assert df["portfolio_growth"].iloc[-1] == pytest.approx(1.06)
    assert df["benchmark_growth"].iloc[-1] == pytest.approx(110 / prices.iloc[3])
    assert df["net_deposits"].iloc[-1] == 1000.0


def test_build_comparison_stops_at_last_nav_date():
    d = days(10)
    prices = pd.Series(np.linspace(100, 110, 10), index=d)
    nav = pd.Series(1000.0, index=d[:6])
    df = build_comparison(nav, pd.Series(dtype=float, index=pd.DatetimeIndex([])), prices)
    assert df.index[-1] == d[5]


def test_performance_stats_matches_hand_calculation():
    r = pd.Series([0.10, -0.10, 0.05])
    s = performance_stats(r, span_days=30)
    growth = 1.10 * 0.90 * 1.05
    assert s["total_return"] == pytest.approx(growth - 1)
    assert s["max_drawdown"] == pytest.approx(0.90 - 1)       # 1.10 -> 0.99
    assert s["cagr"] is None                                   # under a year
    assert s["vol"] == pytest.approx(r.std() * np.sqrt(252))


def test_drawdown_counts_a_loss_from_day_one():
    s = performance_stats(pd.Series([-0.05, -0.05]), span_days=10)
    assert s["max_drawdown"] == pytest.approx(0.95 * 0.95 - 1)


def test_cagr_appears_after_a_year():
    s = performance_stats(pd.Series([0.0] * 5), span_days=730)
    assert s["cagr"] == pytest.approx(0.0)


def test_summarize_reports_dollar_gap():
    d = days(30)
    prices = pd.Series(np.linspace(100, 120, 30), index=d)
    nav = pd.Series(np.linspace(1000, 1300, 30), index=d)
    df = build_comparison(nav, pd.Series(dtype=float, index=pd.DatetimeIndex([])), prices)
    s = summarize(df)
    assert s["nav"] == pytest.approx(1300)
    assert s["benchmark_same_flows"] == pytest.approx(1200)
    assert s["excess_return"] == pytest.approx(0.30 - 0.20)
    assert s["trading_days"] == 30
