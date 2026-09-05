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
        prices.rolling(window).mean()
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
        prices.rolling(short_window).mean()
    )

    long_ma = (
        prices.rolling(long_window).mean()
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
    """

    if prices.empty:
        raise ValueError(
            "Prices cannot be empty."
        )

    if lookback <= 0:
        raise ValueError(
            "Lookback must be greater than zero."
        )

    momentum = prices.pct_change(
        periods=lookback
    )

    signals = (
        momentum > 0
    ).astype(int)

    signals.name = "Signal"

    return signals


def get_strategy(
    strategy_name,
    prices,
    **parameters
):
    """
    Create a trading strategy using
    the selected parameters.

    Parameters
    ----------
    strategy_name : str
        Strategy identifier.

    prices : pandas.Series
        Historical closing prices.

    parameters : dict
        Strategy-specific parameters.

    Returns
    -------
    pandas.Series
        Trading signals.
    """

    if strategy_name == "buy_and_hold":

        return buy_and_hold_strategy(
            prices
        )

    if strategy_name == "moving_average":

        window = parameters.get(
            "window",
            20
        )

        return moving_average_strategy(
            prices,
            window=window
        )

    if strategy_name == "moving_average_crossover":

        short_window = parameters.get(
            "short_window",
            20
        )

        long_window = parameters.get(
            "long_window",
            50
        )

        return moving_average_crossover_strategy(
            prices,
            short_window=short_window,
            long_window=long_window
        )

    if strategy_name == "momentum":

        lookback = parameters.get(
            "lookback",
            20
        )

        return momentum_strategy(
            prices,
            lookback=lookback
        )

    raise ValueError(
        f"Unknown strategy: {strategy_name}"
    )