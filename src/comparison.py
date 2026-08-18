import pandas as pd

from src.backtest import run_backtest
from src.metrics import calculate_metrics
from src.strategies import (
    buy_and_hold_strategy,
    moving_average_strategy,
    moving_average_crossover_strategy,
    momentum_strategy,
)


def compare_strategies(
    prices,
    initial_capital=10_000
):
    """
    Run multiple trading strategies on the
    same price data and compare their performance.

    Parameters
    ----------
    prices : pandas.Series
        Historical closing prices.

    initial_capital : float
        Starting portfolio value.

    Returns
    -------
    pandas.DataFrame
        Comparison of strategy performance.
    """

    if prices.empty:
        raise ValueError(
            "Prices cannot be empty."
        )

    strategies = {
        "Buy & Hold": (
            buy_and_hold_strategy(prices)
        ),

        "Moving Average": (
            moving_average_strategy(
                prices,
                window=20
            )
        ),

        "MA Crossover": (
            moving_average_crossover_strategy(
                prices,
                short_window=20,
                long_window=50
            )
        ),

        "Momentum": (
            momentum_strategy(
                prices,
                lookback=20
            )
        ),
    }

    comparison = []

    for name, signals in strategies.items():

        # Run the backtest
        results = run_backtest(
            prices,
            signals,
            initial_capital=initial_capital
        )

        # Calculate metrics
        metrics = calculate_metrics(
            results["Strategy Return"]
        )

        # Final portfolio value
        final_value = (
            results["Portfolio Value"]
            .iloc[-1]
        )

        comparison.append(
            {
                "Strategy": name,

                "Total Return":
                    metrics["Total Return"],

                "Annualized Return":
                    metrics["Annualized Return"],

                "Volatility":
                    metrics["Volatility"],

                "Sharpe Ratio":
                    metrics["Sharpe Ratio"],

                "Maximum Drawdown":
                    metrics["Maximum Drawdown"],

                "Win Rate":
                    metrics["Win Rate"],

                "Final Portfolio Value":
                    final_value,
            }
        )

    return pd.DataFrame(
        comparison
    )


def print_comparison(
    comparison
):
    """
    Print strategy comparison in a
    readable format.
    """

    display = comparison.copy()

    percentage_columns = [
        "Total Return",
        "Annualized Return",
        "Volatility",
        "Maximum Drawdown",
        "Win Rate",
    ]

    for column in percentage_columns:

        display[column] = (
            display[column] * 100
        ).round(2)

    display["Sharpe Ratio"] = (
        display["Sharpe Ratio"]
        .round(2)
    )

    display["Final Portfolio Value"] = (
        display["Final Portfolio Value"]
        .round(2)
    )

    print()
    print("=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)

    print(
        display.to_string(
            index=False
        )
    )

    print("=" * 80)