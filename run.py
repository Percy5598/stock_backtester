from src.data import get_stock_data
from src.strategies import moving_average_strategy
from src.backtest import run_backtest
from src.metrics import calculate_metrics, print_metrics


# ==================================================
# SETTINGS
# ==================================================

TICKER = "AAPL"
PERIOD = "5y"
INITIAL_CAPITAL = 10_000


# ==================================================
# 1. GET STOCK DATA
# ==================================================

data = get_stock_data(
    TICKER,
    period=PERIOD
)


# ==================================================
# 2. GET CLOSING PRICES
# ==================================================

prices = data["Close"]

# Handle yfinance MultiIndex/DataFrame
if hasattr(prices, "columns"):
    prices = prices.iloc[:, 0]


# ==================================================
# 3. CREATE TRADING SIGNALS
# ==================================================

signals = moving_average_strategy(
    prices,
    window=20
)


# ==================================================
# 4. RUN BACKTEST
# ==================================================

results = run_backtest(
    prices,
    signals,
    initial_capital=INITIAL_CAPITAL
)


# ==================================================
# 5. CALCULATE PERFORMANCE METRICS
# ==================================================

metrics = calculate_metrics(
    results["Strategy Return"]
)


# ==================================================
# 6. PRINT RESULTS
# ==================================================

print()
print("=" * 50)
print("STOCK BACKTEST")
print("=" * 50)

print(f"Ticker: {TICKER}")
print(f"Period: {PERIOD}")
print(
    f"Initial Capital: "
    f"€{INITIAL_CAPITAL:,.2f}"
)

print()

print_metrics(metrics)


# ==================================================
# 7. FINAL PORTFOLIO VALUE
# ==================================================

final_value = (
    results["Portfolio Value"]
    .iloc[-1]
)

print(
    f"Final Portfolio Value: "
    f"€{final_value:,.2f}"
)

print("=" * 50)