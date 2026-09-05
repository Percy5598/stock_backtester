from src.data import get_close_prices
from src.comparison import (
    compare_strategies,
    print_comparison,
)
from src.visualization import plot_equity_curves


TICKER = "AAPL"
PERIOD = "5y"
INITIAL_CAPITAL = 10_000


def main():

    print(
        f"Downloading data for {TICKER}..."
    )

    prices = get_close_prices(
        TICKER,
        period=PERIOD
    )

    strategies = [
        "buy_and_hold",
        "moving_average",
        "moving_average_crossover",
        "momentum",
    ]

    strategy_parameters = {
        "moving_average": {
            "window": 20
        },

        "moving_average_crossover": {
            "short_window": 20,
            "long_window": 50,
        },

        "momentum": {
            "lookback": 20
        },
    }

    comparison, backtest_results = (
        compare_strategies(
            prices,
            initial_capital=INITIAL_CAPITAL,
            strategies=strategies,
            strategy_parameters=strategy_parameters,
        )
    )

    print_comparison(
        comparison
    )

    plot_equity_curves(
        backtest_results,
        title=f"{TICKER} Strategy Performance",
        save_path="equity_curves.png",
    )


if __name__ == "__main__":
    main()