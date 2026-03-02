import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -------------------
# SETTINGS
# -------------------
ticker = "^GSPC"
start = "2026-02-01"
end = "2026-03-01"

# -------------------
# Download Data
# -------------------
data = yf.download(
    ticker,
    start=start,
    end=end,
    auto_adjust=True,
    progress=False,
    threads=False
)

if data is None or data.empty:
    raise ValueError(f"No data returned for {ticker}. Check ticker or date range.")

# Handle MultiIndex safely
if isinstance(data.columns, pd.MultiIndex):
    close_prices = data['Close'].xs(ticker, axis=1)
else:
    close_prices = data['Close']

# -------------------
# Compute Returns
# -------------------
returns = close_prices.pct_change().dropna()
cum_return = (1 + returns).cumprod()

vol = returns.std() * np.sqrt(252)
sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

rolling_max = cum_return.cummax()
drawdown = (cum_return / rolling_max) - 1
mdd = drawdown.min()

# -------------------
# Performance Stats
# -------------------
total_return = cum_return.iloc[-1] - 1
vol = returns.std() * np.sqrt(252)
sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
mdd = ((cum_return / cum_return.cummax()) - 1).min()

stats = (
    f"Return: {total_return:.2%} | "
    f"Vol: {vol:.2%} | "
    f"Sharpe: {sharpe:.2f} | "
    f"Max DD: {mdd:.2%}"
)

# -------------------
# Plot
# -------------------
plt.figure(figsize=(12,6))
plt.plot(cum_return.index, cum_return, label="Cumulative Return")

plt.title(f"{ticker} Performance")
plt.ylabel("Growth of $1")
plt.xlabel("Date")
plt.grid(True, linestyle='--', alpha=0.7)

plt.figtext(0.5, 0.02, stats, ha="center",
            fontsize=10,
            bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig("01_SNP500_performance.png", dpi=300, bbox_inches="tight")

plt.show()