from src.data import get_stock_data

from src.strategies import (
    buy_and_hold_strategy,
    moving_average_strategy,
    moving_average_crossover_strategy,
    momentum_strategy,
)

from src.backtest import run_backtest

from src.comparison import (
    compare_strategies,
    print_comparison,
)

from src.visualization import plot_equity_curves


# ==================================================
# SETTINGS
# ==================================================

TICKER = "AAPL"
PERIOD = "5y"
INITIAL_CAPITAL = 10_000


# ==================================================
# DOWNLOAD DATA
# ==================================================

print(f"Downloading {TICKER} data...")

data = get_stock_data(
    TICKER,
    period=PERIOD
)

prices = data["Close"]

# yfinance can sometimes return
# a DataFrame instead of a Series.

if isinstance(prices, type(data)):
    prices = prices.iloc[:, 0]

print(f"Loaded {len(prices)} price records.")


# ==================================================
# CREATE STRATEGIES
# ==================================================

print("Generating trading signals...")

strategies = {

    "Buy & Hold":
        buy_and_hold_strategy(prices),

    "Moving Average":
        moving_average_strategy(
            prices,
            window=20
        ),

    "MA Crossover":
        moving_average_crossover_strategy(
            prices,
            short_window=20,
            long_window=50
        ),

    "Momentum":
        momentum_strategy(
            prices,
            lookback=20
        ),
}


# ==================================================
# BACKTEST
# ==================================================

print("Running backtests...")

backtest_results = {}

for name, signals in strategies.items():

    print(f"  → {name}")

    results = run_backtest(
        prices,
        signals,
        initial_capital=INITIAL_CAPITAL
    )

    backtest_results[name] = results


# ==================================================
# STRATEGY COMPARISON
# ==================================================

print("\nCalculating performance...")

comparison = compare_strategies(
    prices,
    initial_capital=INITIAL_CAPITAL
)

print_comparison(comparison)


# ==================================================
# VISUALIZATION
# ==================================================

print("\nCreating equity curve...")

plot_equity_curves(
    backtest_results,
    title=f"{TICKER} Strategy Performance",
    save_path="equity_curves.png",
)

print("Done!")