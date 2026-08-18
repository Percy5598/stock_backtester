import pandas as pd
import numpy as np


def run_backtest(
    prices,
    signals,
    initial_capital=10_000
):
    """
    Run a simple long-only backtest.

    Parameters
    ----------
    prices : pandas.Series
        Historical closing prices.

    signals : pandas.Series
        Trading signals:
            1 = invested
            0 = out of market

    initial_capital : float
        Starting portfolio value.

    Returns
    -------
    pandas.DataFrame
        Backtest results containing:

        Price
        Signal
        Daily Return
        Strategy Return
        Portfolio Value
    """

    if prices.empty:
        raise ValueError(
            "Prices cannot be empty."
        )

    if signals.empty:
        raise ValueError(
            "Signals cannot be empty."
        )

    # Make sure both objects use the
    # same dates.
    data = pd.DataFrame(
        {
            "Price": prices,
            "Signal": signals
        }
    ).dropna()

    if data.empty:
        raise ValueError(
            "No overlapping price and signal data."
        )

    # --------------------------------------------------
    # Market returns
    # --------------------------------------------------

    data["Daily Return"] = (
        data["Price"]
        .pct_change()
    )

    # --------------------------------------------------
    # Strategy returns
    #
    # We use yesterday's signal to determine
    # today's position.
    #
    # This is important because using today's
    # signal would introduce look-ahead bias.
    # --------------------------------------------------

    data["Strategy Return"] = (
        data["Daily Return"]
        * data["Signal"].shift(1)
    )

    # First return is undefined.
    data["Strategy Return"] = (
        data["Strategy Return"]
        .fillna(0)
    )

    # --------------------------------------------------
    # Portfolio value
    # --------------------------------------------------

    data["Portfolio Value"] = (
        initial_capital
        * (
            1 + data["Strategy Return"]
        ).cumprod()
    )

    return data


def calculate_backtest_return(
    results,
    initial_capital=10_000
):
    """
    Calculate total strategy return.
    """

    if results.empty:
        return np.nan

    final_value = (
        results["Portfolio Value"]
        .iloc[-1]
    )

    return (
        final_value / initial_capital
        - 1
    )


def calculate_buy_and_hold(
    prices,
    initial_capital=10_000
):
    """
    Calculate the portfolio value of a
    simple buy-and-hold strategy.
    """

    if prices.empty:
        raise ValueError(
            "Prices cannot be empty."
        )

    initial_price = prices.iloc[0]

    portfolio_value = (
        initial_capital
        * prices
        / initial_price
    )

    return portfolio_value