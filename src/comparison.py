import pandas as pd

from src.backtest import run_backtest
from src.metrics import calculate_metrics
from src.strategies import get_strategy


def compare_strategies(
    prices,
    initial_capital=10_000,
    strategies=None,
    strategy_parameters=None,
):
    """
    Run selected strategies on one stock.
    """

    if prices.empty:
        raise ValueError("Prices cannot be empty.")

    if strategies is None:
        strategies = [
            "buy_and_hold",
            "moving_average",
            "moving_average_crossover",
            "momentum",
        ]

    if strategy_parameters is None:
        strategy_parameters = {}

    comparison = []
    backtest_results = {}

    for strategy_name in strategies:

        parameters = strategy_parameters.get(
            strategy_name,
            {}
        )

        signals = get_strategy(
            strategy_name,
            prices,
            **parameters
        )

        results = run_backtest(
            prices,
            signals,
            initial_capital=initial_capital,
        )

        backtest_results[strategy_name] = results

        metrics = calculate_metrics(
            results["Strategy Return"]
        )

        final_value = (
            results["Portfolio Value"].iloc[-1]
        )

        comparison.append(
            {
                "Strategy": strategy_name,
                "Total Return": metrics["Total Return"],
                "Annualized Return": metrics["Annualized Return"],
                "Volatility": metrics["Volatility"],
                "Sharpe Ratio": metrics["Sharpe Ratio"],
                "Maximum Drawdown": metrics["Maximum Drawdown"],
                "Win Rate": metrics["Win Rate"],
                "Final Portfolio Value": final_value,
            }
        )

    comparison = pd.DataFrame(comparison)

    return comparison, backtest_results


def compare_multiple_stocks(
    stock_prices,
    initial_capital=10_000,
    strategies=None,
    strategy_parameters=None,
):
    """
    Run selected strategies across multiple stocks.

    Parameters
    ----------
    stock_prices : dict
        Example:
        {
            "AAPL": prices,
            "MSFT": prices,
            "NVDA": prices
        }

    Returns
    -------
    comparison : pandas.DataFrame
    backtest_results : dict
    """

    if not stock_prices:
        raise ValueError(
            "Stock prices cannot be empty."
        )

    all_comparisons = {}
    all_backtest_results = {}

    for ticker, prices in stock_prices.items():

        if prices.empty:
            continue

        comparison, results = compare_strategies(
            prices,
            initial_capital=initial_capital,
            strategies=strategies,
            strategy_parameters=strategy_parameters,
        )

        comparison.insert(
            0,
            "Ticker",
            ticker
        )

        all_comparisons[ticker] = comparison

        all_backtest_results[ticker] = results

    if not all_comparisons:
        raise ValueError(
            "No valid stock data available."
        )

    comparison_df = pd.concat(
        all_comparisons.values(),
        ignore_index=True
    )

    return (
        comparison_df,
        all_backtest_results
    )


def format_comparison(comparison):
    """
    Format comparison results for display.
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

        if column in display.columns:

            display[column] = (
                display[column] * 100
            ).round(2)

    if "Sharpe Ratio" in display.columns:

        display["Sharpe Ratio"] = (
            display["Sharpe Ratio"]
            .round(2)
        )

    if "Final Portfolio Value" in display.columns:

        display["Final Portfolio Value"] = (
            display["Final Portfolio Value"]
            .round(2)
        )

    return display


def print_comparison(comparison):
    """
    Print comparison results.
    """

    display = format_comparison(
        comparison
    )

    print()
    print("=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)

    print(
        display.to_string(
            index=False
        )
    )

    print("=" * 100)