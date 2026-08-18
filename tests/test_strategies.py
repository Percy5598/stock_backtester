import pandas as pd
import pytest

from src.strategies import (
    buy_and_hold_strategy,
    moving_average_strategy,
    moving_average_crossover_strategy,
    momentum_strategy,
    get_strategy,
)


# ==================================================
# TEST DATA
# ==================================================

@pytest.fixture
def prices():
    """
    Simple deterministic price series
    used for testing strategies.
    """

    return pd.Series(
        [
            100,
            102,
            104,
            106,
            108,
            110,
            108,
            106,
            104,
            102,
        ],
        index=pd.date_range(
            "2026-01-01",
            periods=10,
            freq="D"
        ),
        name="Close"
    )


# ==================================================
# BUY & HOLD
# ==================================================

def test_buy_and_hold_strategy(prices):

    signals = buy_and_hold_strategy(
        prices
    )

    # Buy & hold should always
    # remain invested.
    assert (signals == 1).all()

    # Signal should have the same
    # index as the price data.
    assert signals.index.equals(
        prices.index
    )

    # Number of signals should equal
    # number of prices.
    assert len(signals) == len(prices)


# ==================================================
# MOVING AVERAGE STRATEGY
# ==================================================

def test_moving_average_strategy(prices):

    signals = moving_average_strategy(
        prices,
        window=3
    )

    # Signals should only contain
    # 0 or 1.
    assert set(
        signals.dropna().unique()
    ).issubset({0, 1})

    # Signal index should match
    # price index.
    assert signals.index.equals(
        prices.index
    )

    # Once enough data exists,
    # there should be valid signals.
    assert signals.iloc[2:].notna().all()


# ==================================================
# MOVING AVERAGE STRATEGY LOGIC
# ==================================================

def test_moving_average_logic():

    prices = pd.Series(
        [
            100,
            100,
            100,
            120,
        ]
    )

    signals = moving_average_strategy(
        prices,
        window=3
    )

    # Last three prices:
    #
    # 100, 100, 120
    #
    # Moving average = 106.67
    #
    # Price = 120
    #
    # 120 > 106.67
    #
    # Therefore signal = 1

    assert signals.iloc[-1] == 1


# ==================================================
# MOVING AVERAGE CROSSOVER
# ==================================================

def test_moving_average_crossover(prices):

    signals = (
        moving_average_crossover_strategy(
            prices,
            short_window=2,
            long_window=4
        )
    )

    # Signals should only contain
    # 0 or 1.
    assert set(
        signals.dropna().unique()
    ).issubset({0, 1})

    assert signals.index.equals(
        prices.index
    )


# ==================================================
# CROSSOVER LOGIC
# ==================================================

def test_crossover_logic():

    prices = pd.Series(
        [
            100,
            100,
            100,
            100,
            120,
        ]
    )

    signals = (
        moving_average_crossover_strategy(
            prices,
            short_window=2,
            long_window=4
        )
    )

    # At the final point:
    #
    # Short MA:
    # (100 + 120) / 2 = 110
    #
    # Long MA:
    # (100 + 100 + 100 + 120) / 4
    # = 105
    #
    # 110 > 105
    #
    # Signal should be 1.

    assert signals.iloc[-1] == 1


# ==================================================
# MOMENTUM STRATEGY
# ==================================================

def test_momentum_strategy(prices):

    signals = momentum_strategy(
        prices,
        lookback=3
    )

    # Signals should only contain
    # 0 or 1.
    assert set(
        signals.dropna().unique()
    ).issubset({0, 1})

    assert signals.index.equals(
        prices.index
    )


# ==================================================
# POSITIVE MOMENTUM
# ==================================================

def test_positive_momentum():

    prices = pd.Series(
        [
            100,
            105,
            110,
            120,
        ]
    )

    signals = momentum_strategy(
        prices,
        lookback=2
    )

    # Last two-period return:
    #
    # 120 / 105 - 1 > 0
    #
    # Therefore signal = 1.

    assert signals.iloc[-1] == 1


# ==================================================
# NEGATIVE MOMENTUM
# ==================================================

def test_negative_momentum():

    prices = pd.Series(
        [
            120,
            110,
            100,
            90,
        ]
    )

    signals = momentum_strategy(
        prices,
        lookback=2
    )

    # Last two-period return is negative.
    #
    # Therefore signal = 0.

    assert signals.iloc[-1] == 0


# ==================================================
# GET STRATEGY
# ==================================================

def test_get_strategy(prices):

    strategy_names = [
        "buy_and_hold",
        "moving_average",
        "moving_average_crossover",
        "momentum",
    ]

    for name in strategy_names:

        signals = get_strategy(
            name,
            prices
        )

        assert isinstance(
            signals,
            pd.Series
        )

        assert signals.index.equals(
            prices.index
        )


# ==================================================
# INVALID STRATEGY
# ==================================================

def test_invalid_strategy(prices):

    with pytest.raises(
        ValueError
    ):

        get_strategy(
            "does_not_exist",
            prices
        )


# ==================================================
# INVALID MOVING AVERAGE WINDOW
# ==================================================

def test_invalid_moving_average_window(
    prices
):

    with pytest.raises(
        ValueError
    ):

        moving_average_strategy(
            prices,
            window=0
        )


# ==================================================
# INVALID CROSSOVER WINDOWS
# ==================================================

def test_invalid_crossover_windows(
    prices
):

    with pytest.raises(
        ValueError
    ):

        moving_average_crossover_strategy(
            prices,
            short_window=50,
            long_window=20
        )


# ==================================================
# INVALID MOMENTUM LOOKBACK
# ==================================================

def test_invalid_momentum_lookback(
    prices
):

    with pytest.raises(
        ValueError
    ):

        momentum_strategy(
            prices,
            lookback=0
        )


# ==================================================
# EMPTY DATA
# ==================================================

def test_empty_prices():

    prices = pd.Series(
        dtype=float
    )

    with pytest.raises(
        ValueError
    ):

        buy_and_hold_strategy(
            prices
        )

    with pytest.raises(
        ValueError
    ):

        moving_average_strategy(
            prices
        )

    with pytest.raises(
        ValueError
    ):

        momentum_strategy(
            prices
        )