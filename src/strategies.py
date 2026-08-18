import pandas as pd


def buy_and_hold_strategy(prices):
    """
    Buy and hold the stock for the entire period.

    Signal:
        1 = invested
    """

    if prices.empty:
        raise ValueError(
            "Prices cannot be empty."
        )

    signals = pd.Series(
        1,
        index=prices.index,
        name="Signal"
    )

    return signals


def moving_average_strategy(
    prices,
    window=20
):
    """
    Simple moving-average strategy.

    Rule:
        Price > moving average -> 1
        Price <= moving average -> 0

    Parameters
    ----------
    prices : pandas.Series
        Historical closing prices.

    window : int
        Moving-average window.
        Default = 20 trading days.

    Returns
    -------
    pandas.Series
        Trading signals.
    """

    if prices.empty:
        raise ValueError(
            "Prices cannot be empty."
        )

    if window <= 0:
        raise ValueError(
            "Window must be greater than zero."
        )

    moving_average = (
        prices
        .rolling(window)
        .mean()
    )

    signals = (
        prices > moving_average
    ).astype(int)

    signals.name = "Signal"

    return signals


def moving_average_crossover_strategy(
    prices,
    short_window=20,
    long_window=50
):
    """
    Moving-average crossover strategy.

    Rule:
        Short MA > Long MA -> 1
        Short MA <= Long MA -> 0

    Example:
        20-day MA vs 50-day MA.

    Parameters
    ----------
    prices : pandas.Series
        Historical closing prices.

    short_window : int
        Short moving-average window.

    long_window : int
        Long moving-average window.

    Returns
    -------
    pandas.Series
        Trading signals.
    """

    if prices.empty:
        raise ValueError(
            "Prices cannot be empty."
        )

    if short_window <= 0:
        raise ValueError(
            "Short window must be greater than zero."
        )

    if long_window <= 0:
        raise ValueError(
            "Long window must be greater than zero."
        )

    if short_window >= long_window:
        raise ValueError(
            "Short window must be smaller "
            "than long window."
        )

    short_ma = (
        prices
        .rolling(short_window)
        .mean()
    )

    long_ma = (
        prices
        .rolling(long_window)
        .mean()
    )

    signals = (
        short_ma > long_ma
    ).astype(int)

    signals.name = "Signal"

    return signals


def momentum_strategy(
    prices,
    lookback=20
):
    """
    Simple momentum strategy.

    Rule:
        Positive lookback return -> 1
        Negative lookback return -> 0

    Example:
        If the stock increased over the
        previous 20 trading days, stay invested.
    """

    if prices.empty:
        raise ValueError(
            "Prices cannot be empty."
        )

    if lookback <= 0:
        raise ValueError(
            "Lookback must be greater than zero."
        )

    momentum = (
        prices.pct_change(
            periods=lookback
        )
    )

    signals = (
        momentum > 0
    ).astype(int)

    signals.name = "Signal"

    return signals


def get_strategy(
    strategy_name,
    prices
):
    """
    Select a strategy by name.

    Available strategies:

        "buy_and_hold"
        "moving_average"
        "moving_average_crossover"
        "momentum"
    """

    if strategy_name == "buy_and_hold":

        return buy_and_hold_strategy(
            prices
        )

    elif strategy_name == "moving_average":

        return moving_average_strategy(
            prices,
            window=20
        )

    elif strategy_name == "moving_average_crossover":

        return moving_average_crossover_strategy(
            prices,
            short_window=20,
            long_window=50
        )

    elif strategy_name == "momentum":

        return momentum_strategy(
            prices,
            lookback=20
        )

    else:

        raise ValueError(
            f"Unknown strategy: "
            f"{strategy_name}"
        )