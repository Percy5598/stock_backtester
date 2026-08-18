import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    calculate_total_return,
    calculate_annualized_return,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_drawdown_series,
    calculate_win_rate,
    calculate_average_return,
    calculate_metrics,
)


# ==================================================
# TOTAL RETURN
# ==================================================

def test_total_return():
    returns = pd.Series(
        [0.10, 0.10]
    )

    result = calculate_total_return(
        returns
    )

    # 1.10 * 1.10 - 1 = 0.21
    assert result == pytest.approx(
        0.21
    )


# ==================================================
# ANNUALIZED RETURN
# ==================================================

def test_annualized_return():
    returns = pd.Series(
        [0.01] * 252
    )

    result = calculate_annualized_return(
        returns
    )

    assert result > 0


# ==================================================
# VOLATILITY
# ==================================================

def test_volatility():
    returns = pd.Series(
        [0.01, -0.01] * 100
    )

    result = calculate_volatility(
        returns
    )

    assert result > 0


# ==================================================
# SHARPE RATIO
# ==================================================

def test_sharpe_ratio():
    returns = pd.Series(
        [0.01] * 252
    )

    result = calculate_sharpe_ratio(
        returns,
        risk_free_rate=0
    )

    assert result > 0


# ==================================================
# MAXIMUM DRAWDOWN
# ==================================================

def test_max_drawdown():
    returns = pd.Series(
        [
            0.10,
            0.10,
            -0.20,
            0.05,
        ]
    )

    result = calculate_max_drawdown(
        returns
    )

    # After +10%, +10%:
    #
    # 1.00 → 1.10 → 1.21
    #
    # Then -20%:
    #
    # 1.21 → 0.968
    #
    # Drawdown:
    #
    # 0.968 / 1.21 - 1
    # = -0.20
    #
    assert result == pytest.approx(
        -0.20
    )


# ==================================================
# DRAWDOWN SERIES
# ==================================================

def test_drawdown_series():
    returns = pd.Series(
        [
            0.10,
            0.10,
            -0.20,
        ]
    )

    result = calculate_drawdown_series(
        returns
    )

    assert result.iloc[0] == pytest.approx(
        0
    )

    assert result.iloc[-1] == pytest.approx(
        -0.20
    )


# ==================================================
# WIN RATE
# ==================================================

def test_win_rate():
    returns = pd.Series(
        [
            0.10,
            -0.05,
            0.03,
            -0.02,
        ]
    )

    result = calculate_win_rate(
        returns
    )

    # 2 positive returns / 4 total
    assert result == pytest.approx(
        0.50
    )


# ==================================================
# AVERAGE RETURN
# ==================================================

def test_average_return():
    returns = pd.Series(
        [
            0.10,
            0.20,
            -0.10,
        ]
    )

    result = calculate_average_return(
        returns
    )

    expected = (
        0.10
        + 0.20
        - 0.10
    ) / 3

    assert result == pytest.approx(
        expected
    )


# ==================================================
# COMPLETE METRICS
# ==================================================

def test_calculate_metrics():
    returns = pd.Series(
        [
            0.01,
            0.02,
            -0.01,
            0.03,
            -0.02,
        ]
    )

    result = calculate_metrics(
        returns
    )

    assert isinstance(
        result,
        dict
    )

    assert "Total Return" in result
    assert "Annualized Return" in result
    assert "Volatility" in result
    assert "Sharpe Ratio" in result
    assert "Maximum Drawdown" in result
    assert "Win Rate" in result
    assert "Average Daily Return" in result


# ==================================================
# EMPTY DATA
# ==================================================

def test_empty_returns():
    returns = pd.Series(
        dtype=float
    )

    assert np.isnan(
        calculate_total_return(
            returns
        )
    )

    assert np.isnan(
        calculate_volatility(
            returns
        )
    )

    assert np.isnan(
        calculate_sharpe_ratio(
            returns
        )
    )

    assert np.isnan(
        calculate_max_drawdown(
            returns
        )
    )


# ==================================================
# NaN VALUES
# ==================================================
def test_nan_returns_are_ignored():
    returns = pd.Series(
        [
            0.10,
            np.nan,
            0.20,
        ]
    )

    result = calculate_total_return(
        returns
    )

    expected = (
        1.10
        * 1.20
        - 1
    )

    assert result == pytest.approx(
        expected
    )