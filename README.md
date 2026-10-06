# miltonia-research

## IBKR portfolio vs S&P 500

`backtesting/02_IBKR_vs_SNP500.py` pulls your account history from Interactive Brokers and compares it with the S&P 500 from the day you first had money invested to the latest report date.

```
pip install -r requirements.txt
python backtesting/02_IBKR_vs_SNP500.py            # cached IBKR data is reused for 12h
python backtesting/02_IBKR_vs_SNP500.py --refresh  # force a fresh download
```

It prints a summary and writes a chart and a daily table to `backtesting/output/`. That folder, the IBKR cache in `backtesting/data/`, and `.env` are git-ignored because they contain your balances and token.

### One-time setup: Flex Query + token

The script reads your history through IBKR's [Flex Web Service](https://www.interactivebrokers.com/campus/ibkr-api-page/flex-web-service/) (read-only, no TWS/Gateway needed). You create the query and token in Client Portal yourself; the token never needs to be shared with anyone but this script.

1. **Performance & Reports → Flex Queries → Activity Flex Query → Create.** Add two sections:
   - **Net Asset Value (NAV) in Base**: tick **Report Date** and **Total** (add Currency if offered); turn on **Breakout by Day**.
   - **Cash Transactions**: tick **Type, Amount, Currency, FX Rate To Base, Report Date, Date/Time**, and the **Detail** level (not Summary).
2. In the query's **Delivery Configuration**: Format **XML**, Date Format **yyyyMMdd**, Time Format **HHmmss**, separator **;** . The period doesn't matter, the script overrides it.
3. Save it and note the **Query ID** shown in the list.
4. **Performance & Reports → Flex Queries → Flex Web Service Configuration** (gear icon): enable it and generate a **token**. Treat it like a password.
5. `copy .env.example .env` and fill in `IBKR_FLEX_TOKEN` and `IBKR_FLEX_QUERY_ID`.

IBKR's field labels shift occasionally. If a field is missing, the script stops with a message naming the one to add.

### How it's calculated

- **Return** is time-weighted: each day's return is `(NAV - deposits that day) / previous NAV - 1`, with deposits assumed to land at the close, so adding or withdrawing cash never counts as performance. Vol, Sharpe (0% risk-free) and max drawdown use the same daily series as `01_SNP500.py`.
- **Benchmark** is SPY with dividends reinvested, since your IBKR return includes dividends. Set `benchmark = "^GSPC"` for the price-only index.
- **"Same deposits in SPY"** invests every deposit (and sells for every withdrawal) in the benchmark at that day's close, so the dollar gap to your actual balance is like for like.
- The period is detected from your first funded day. Set `start` in the script to override it.

### Caveats

- Securities moved in from another broker (ACATS/in-kind) aren't cash deposits, so they show up as a one-day jump in return. The script warns about daily moves over 10%.
- If your IBKR base currency isn't USD, currency moves are in your return but not in SPY's.
- Flex data lags to the last business day.

### Tests

```
python -m pytest tests
```
