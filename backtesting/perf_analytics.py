"""Portfolio-vs-benchmark maths: time-weighted returns that ignore deposits and
withdrawals, plus "what if every deposit had gone into the benchmark" value."""
import numpy as np
import pandas as pd


def align_to_trading_days(nav, flows, days):
    """NAV and net deposits on the benchmark's trading days.

    A deposit dated on a non-trading day rolls into the next trading day.
    """
    nav_d = nav.reindex(nav.index.union(days)).ffill().reindex(days)
    pos = days.searchsorted(flows.index)        # first trading day >= flow date
    keep = pos < len(days)
    flows_d = (
        pd.Series(flows.to_numpy()[keep], index=days[pos[keep]])
        .groupby(level=0).sum()
        .reindex(days, fill_value=0.0)
    )
    return nav_d, flows_d


def twr_returns(nav, flows):
    """Daily time-weighted returns, with each flow assumed to land at day end:

        r_t = (NAV_t - flow_t) / NAV_{t-1} - 1

    so a deposit never counts as performance. The first day is the base (no return).
    """
    prev = nav.shift(1)
    r = (nav - flows) / prev.where(prev > 0) - 1
    return r.iloc[1:].fillna(0.0)


def benchmark_with_same_flows(prices, flows, start_value):
    """Value of a portfolio that starts with `start_value` in the benchmark and
    buys (deposit) or sells (withdrawal) it at that day's close."""
    flow = flows.copy()
    flow.iloc[0] = 0.0
    units = start_value / prices.iloc[0] + (flow / prices).cumsum()
    return units * prices


def build_comparison(nav, flows, prices):
    """Daily portfolio vs benchmark table from inception to the last NAV date."""
    funded = nav.gt(0).to_numpy()
    if not funded.any():
        raise ValueError("Account NAV is never positive - nothing to benchmark.")
    nav = nav.iloc[funded.argmax():]
    days = prices.index[(prices.index >= nav.index[0]) & (prices.index <= nav.index[-1])]
    if len(days) < 2:
        raise ValueError("Need at least two benchmark trading days within the invested period.")

    prices = prices.loc[days]
    nav_d, flows_d = align_to_trading_days(nav, flows, days)

    portfolio_ret = twr_returns(nav_d, flows_d).reindex(days)
    benchmark_ret = prices.pct_change()
    contributed = flows_d.copy()
    contributed.iloc[0] = nav_d.iloc[0]

    return pd.DataFrame({
        "nav": nav_d,
        "flow": flows_d,
        "net_deposits": contributed.cumsum(),
        "portfolio_return": portfolio_ret,
        "benchmark_return": benchmark_ret,
        "portfolio_growth": (1 + portfolio_ret.fillna(0.0)).cumprod(),
        "benchmark_growth": (1 + benchmark_ret.fillna(0.0)).cumprod(),
        "benchmark_same_flows": benchmark_with_same_flows(prices, flows_d, nav_d.iloc[0]),
    })


def performance_stats(returns, span_days, periods=252):
    """Same measures as 01_SNP500.py (Sharpe uses a 0% risk-free rate), plus CAGR
    once there is at least a year of history."""
    growth = (1 + returns).cumprod()
    total_return = growth.iloc[-1] - 1
    sd = returns.std()
    peak = np.maximum(growth.cummax(), 1.0)
    return {
        "total_return": total_return,
        "cagr": (1 + total_return) ** (365.25 / span_days) - 1 if span_days >= 365 else None,
        "vol": sd * np.sqrt(periods),
        "sharpe": returns.mean() / sd * np.sqrt(periods) if sd > 0 else float("nan"),
        "max_drawdown": (growth / peak - 1).min(),
    }


def summarize(df, min_obs_for_beta=20):
    """Headline numbers for the comparison table."""
    p = df["portfolio_return"].dropna()
    b = df["benchmark_return"].dropna()
    span_days = (df.index[-1] - df.index[0]).days
    portfolio = performance_stats(p, span_days)
    benchmark = performance_stats(b, span_days)

    beta = correlation = float("nan")
    if len(p) >= min_obs_for_beta and b.var() > 0:
        beta = p.cov(b) / b.var()
        correlation = p.corr(b)

    last = df.iloc[-1]
    return {
        "start": df.index[0],
        "end": df.index[-1],
        "trading_days": len(df),
        "portfolio": portfolio,
        "benchmark": benchmark,
        "excess_return": portfolio["total_return"] - benchmark["total_return"],
        "beta": beta,
        "correlation": correlation,
        "nav": last["nav"],
        "net_deposits": last["net_deposits"],
        "benchmark_same_flows": last["benchmark_same_flows"],
    }
