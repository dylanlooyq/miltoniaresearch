"""Benchmark your IBKR portfolio against the S&P 500 for as long as you've had money invested.

    python backtesting/02_IBKR_vs_SNP500.py             # uses cached IBKR data if < 12h old
    python backtesting/02_IBKR_vs_SNP500.py --refresh   # re-download from IBKR

Needs a Flex Query token in .env - see README.md.
"""
import argparse
from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import FuncFormatter, PercentFormatter

import ibkr_flex
from perf_analytics import build_comparison, summarize

# -------------------
# SETTINGS
# -------------------
# SPY = S&P 500 with dividends reinvested, comparable to your IBKR return (which
# includes dividends). Use "^GSPC" for the price-only index.
benchmark = "SPY"
# First day you had money invested. None = detect it from your IBKR history.
start = None

HERE = Path(__file__).resolve().parent
data_dir = HERE / "data"        # cached IBKR data (git-ignored)
output_dir = HERE / "output"    # chart + daily table (git-ignored)

bench_label = "S&P 500" if benchmark == "^GSPC" else f"S&P 500 ({benchmark})"
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "SGD": "S$"}

# Chart colours: accent for the portfolio, de-emphasised gray for the benchmark.
SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, STATS_BG = "#e1e0d9", "#c3c2b7", "#f0efec"
PORTFOLIO_COLOR, BENCHMARK_COLOR = "#2a78d6", MUTED


# -------------------
# Load Data
# -------------------
def load_history(refresh=False):
    if not refresh:
        cached = ibkr_flex.load_cache(data_dir)
        if cached is not None:
            print("Using cached IBKR data (--refresh to re-download).")
            return cached
    token, query_id = ibkr_flex.credentials()
    print("Downloading account history from IBKR Flex (can take a minute)...")
    history = ibkr_flex.fetch_history(
        token, query_id, start=date.fromisoformat(start) if start else None
    )
    ibkr_flex.save_cache(history, data_dir)
    return history


def download_benchmark(ticker, first_day, last_day):
    data = yf.download(
        ticker,
        start=first_day,
        end=last_day + pd.Timedelta(days=1),   # yfinance's end is exclusive
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker or date range.")

    # Handle MultiIndex safely
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data['Close'].xs(ticker, axis=1)
    else:
        close_prices = data['Close']
    close_prices.index = pd.to_datetime(close_prices.index).tz_localize(None).normalize()
    return close_prices.dropna()


# -------------------
# Console summary
# -------------------
def stats_line(label, s):
    cagr = f" | CAGR: {s['cagr']:.2%}" if s["cagr"] is not None else ""
    return (
        f"{label:<22}Return: {s['total_return']:+.2%}{cagr} | Vol: {s['vol']:.2%} | "
        f"Sharpe: {s['sharpe']:.2f} | Max DD: {s['max_drawdown']:.2%}"
    )


def print_summary(df, summary, currency):
    sym = CURRENCY_SYMBOLS.get(currency, currency + " ")
    print(f"\nInvested {summary['start']:%Y-%m-%d} -> {summary['end']:%Y-%m-%d} "
          f"({summary['trading_days']} trading days)\n")
    print(stats_line("Portfolio", summary["portfolio"]))
    print(stats_line(bench_label, summary["benchmark"]))
    print(f"\nExcess return: {summary['excess_return'] * 100:+.2f} pts | "
          f"Beta: {summary['beta']:.2f} | Correlation: {summary['correlation']:.2f}")
    gap = summary["nav"] - summary["benchmark_same_flows"]
    print(f"\nPortfolio value:                   {sym}{summary['nav']:>12,.0f}")
    print(f"Same deposits in {bench_label}: {sym}{summary['benchmark_same_flows']:>12,.0f}")
    print(f"Net deposits:                      {sym}{summary['net_deposits']:>12,.0f}")
    print(f"{'Ahead of' if gap >= 0 else 'Behind'} the index by {sym}{abs(gap):,.0f}")

    if currency != "USD":
        print(f"\nNote: your account is in {currency} but {benchmark} is in USD, so currency "
              f"moves are in your return and not in the benchmark's.")
    jumps = df.index[df["portfolio_return"].abs() > 0.10]
    if len(jumps):
        print("\nWarning: daily moves above 10% on "
              + ", ".join(f"{d:%Y-%m-%d}" for d in jumps[:5])
              + ". Usually an unrecorded transfer or a deposit dated a day off - "
              "check the Flex Query's cash/transfer sections.")


# -------------------
# Plot
# -------------------
class DataDateFormatter(mdates.ConciseDateFormatter):
    """Concise date labels, blank past the last data point (the margin holds end labels)."""

    def __init__(self, locator, last):
        super().__init__(locator)
        self.last = mdates.date2num(last)

    def format_ticks(self, values):
        labels = super().format_ticks(values)
        return [label if v <= self.last else "" for v, label in zip(values, labels)]


def style_axes(ax, last):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DataDateFormatter(locator, last))


def label_ends(ax, items, min_gap_pt, offset_pt=14):
    """Direct end labels. items: (text, x, y, dot_color or None).

    Labels that would overlap are spread apart and tied back to their line end
    with a leader line.
    """
    pt_per_px = 72 / ax.figure.dpi
    ys = [ax.transData.transform((mdates.date2num(x), y))[1] * pt_per_px for _, x, y, _ in items]
    placed, prev = {}, None
    for i in sorted(range(len(items)), key=lambda i: -ys[i]):
        placed[i] = ys[i] if prev is None else min(ys[i], prev - min_gap_pt)
        prev = placed[i]
    for i, (text, x, y, dot_color) in enumerate(items):
        dy = placed[i] - ys[i]
        leader = dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=2, shrinkB=6)
        ax.annotate(text, xy=(x, y), xytext=(offset_pt, dy), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color=INK, linespacing=1.3,
                    arrowprops=leader if abs(dy) > 0.5 else None, annotation_clip=False)
        if dot_color:
            ax.plot(x, y, "o", ms=8, color=dot_color, markeredgecolor=SURFACE,
                    markeredgewidth=2, clip_on=False, zorder=5)


def plot(df, summary, currency, out_png):
    sym = CURRENCY_SYMBOLS.get(currency, currency + " ")
    money = FuncFormatter(lambda v, _: f"{sym}{v:,.0f}")
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig = plt.figure(figsize=(12, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.3,
                          left=0.075, right=0.98, top=0.83, bottom=0.17)
    top = fig.add_subplot(gs[0])
    bottom = fig.add_subplot(gs[1], sharex=top)

    fig.text(0.075, 0.955, f"IBKR portfolio vs {bench_label}", fontsize=17, fontweight="semibold", va="top")
    fig.text(0.075, 0.915,
             f"Since you started investing, {summary['start']:%d %b %Y} to {summary['end']:%d %b %Y}. "
             "Returns are time-weighted, so deposits and withdrawals don't count as performance.",
             fontsize=10, color=SECONDARY, va="top")

    # Panel 1: return since inception
    p_ret = df["portfolio_growth"] - 1
    b_ret = df["benchmark_growth"] - 1
    top.axhline(0, color=BASELINE, linewidth=1)
    top.plot(df.index, b_ret, color=BENCHMARK_COLOR, linewidth=2, solid_capstyle="round", label=bench_label)
    top.plot(df.index, p_ret, color=PORTFOLIO_COLOR, linewidth=2, solid_capstyle="round", label="Portfolio")
    top.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    top.set_title("Return since inception", loc="left", fontsize=11, pad=10)
    top.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=SECONDARY, handlelength=1.6)

    # Panel 2: dollars - what you have vs the same deposits in the index
    bottom.step(df.index, df["net_deposits"], where="post", color=SECONDARY,
                linewidth=1.5, linestyle=(0, (2, 2)), label="Net deposits")
    bottom.plot(df.index, df["benchmark_same_flows"], color=BENCHMARK_COLOR, linewidth=2,
                solid_capstyle="round", label=f"Same deposits in {bench_label}")
    bottom.plot(df.index, df["nav"], color=PORTFOLIO_COLOR, linewidth=2,
                solid_capstyle="round", label="Portfolio")
    bottom.yaxis.set_major_formatter(money)
    bottom.set_title(f"Portfolio value vs the same deposits in {bench_label}", loc="left", fontsize=11, pad=10)
    bottom.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=SECONDARY, handlelength=1.6)

    last, span = df.index[-1], df.index[-1] - df.index[0]
    for ax in (top, bottom):
        style_axes(ax, last)
    top.tick_params(labelbottom=False)

    # Room on the right for end labels, then freeze limits so label positions are exact.
    top.set_xlim(df.index[0], last + span * 0.30)
    for ax in (top, bottom):
        ax.set_ylim(ax.get_ylim())

    label_ends(top, [
        (f"Portfolio\n{p_ret.iloc[-1]:+.1%}", last, p_ret.iloc[-1], PORTFOLIO_COLOR),
        (f"{bench_label}\n{b_ret.iloc[-1]:+.1%}", last, b_ret.iloc[-1], BENCHMARK_COLOR),
    ], min_gap_pt=30)
    label_ends(bottom, [
        (f"Portfolio  {sym}{df['nav'].iloc[-1]:,.0f}", last, df["nav"].iloc[-1], PORTFOLIO_COLOR),
        (f"Same deposits in {benchmark}  {sym}{df['benchmark_same_flows'].iloc[-1]:,.0f}",
         last, df["benchmark_same_flows"].iloc[-1], BENCHMARK_COLOR),
        (f"Net deposits  {sym}{df['net_deposits'].iloc[-1]:,.0f}", last, df["net_deposits"].iloc[-1], None),
    ], min_gap_pt=17)

    stats = (
        stats_line("Portfolio", summary["portfolio"]) + "\n"
        + stats_line(bench_label, summary["benchmark"]) + "\n"
        + f"{'Versus':<22}Excess: {summary['excess_return'] * 100:+.2f} pts | "
          f"Beta: {summary['beta']:.2f} | Correlation: {summary['correlation']:.2f}"
    )
    fig.text(0.075, 0.075, stats, fontsize=9, family=["Consolas", "DejaVu Sans Mono"], va="center",
             linespacing=1.6, bbox={"facecolor": STATS_BG, "edgecolor": "none", "pad": 9})

    fig.savefig(out_png, dpi=300)
    return fig


# -------------------
# Run
# -------------------
def run(history, out_dir=output_dir, prices=None):
    nav = history.nav
    funded = nav[nav > 0]
    if funded.empty:
        raise ValueError("Account NAV is never positive - nothing to benchmark.")
    if prices is None:
        prices = download_benchmark(benchmark, funded.index[0].date(), nav.index[-1].date())

    df = build_comparison(nav, history.flows, prices)
    summary = summarize(df)
    print_summary(df, summary, history.currency)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "02_IBKR_vs_SNP500_daily.csv", index_label="date")
    fig = plot(df, summary, history.currency, out_dir / "02_IBKR_vs_SNP500.png")
    print(f"\nSaved chart and daily table to {out_dir}")
    return df, summary, fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-download from IBKR instead of using the cache")
    parser.add_argument("--no-show", action="store_true", help="save the chart without opening a window")
    args = parser.parse_args()

    run(load_history(refresh=args.refresh))
    if not args.no_show:
        plt.show()
