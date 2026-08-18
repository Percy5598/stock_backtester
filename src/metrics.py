import numpy as np
import pandas as pd


TRADING_DAYS = 252


def calculate_total_return(returns):
    """
    Calculate total compounded return.

    Parameters
    ----------
    returns : pandas.Series
        Periodic returns expressed as decimals.
        Example: 0.05 = 5%

    Returns
    -------
    float
        Total return as a decimal.
    """

    returns = returns.dropna()

    if returns.empty:
        return np.nan

    return (1 + returns).prod() - 1


def calculate_annualized_return(
    returns,
    periods_per_year=TRADING_DAYS
):
    """
    Calculate annualized return.
    """

    returns = returns.dropna()

    if returns.empty:
        return np.nan

    total_return = (
        calculate_total_return(returns)
    )

    number_of_periods = len(returns)

    if number_of_periods == 0:
        return np.nan

    years = (
        number_of_periods
        / periods_per_year
    )

    if years <= 0:
        return np.nan

    return (
        (1 + total_return) ** (1 / years)
    ) - 1


def calculate_volatility(
    returns,
    periods_per_year=TRADING_DAYS
):
    """
    Calculate annualized volatility.

    For daily returns, the standard deviation
    is multiplied by sqrt(252).
    """

    returns = returns.dropna()

    if len(returns) < 2:
        return np.nan

    return (
        returns.std()
        * np.sqrt(periods_per_year)
    )


def calculate_sharpe_ratio(
    returns,
    risk_free_rate=0.02,
    periods_per_year=TRADING_DAYS
):
    """
    Calculate annualized Sharpe ratio.

    Parameters
    ----------
    returns : pandas.Series
        Daily strategy returns.

    risk_free_rate : float
        Annual risk-free rate.
        Default = 2%.

    periods_per_year : int
        Number of trading periods per year.
        Default = 252.
    """

    returns = returns.dropna()

    if len(returns) < 2:
        return np.nan

    daily_risk_free_rate = (
        (1 + risk_free_rate)
        ** (1 / periods_per_year)
        - 1
    )

    excess_returns = (
        returns
        - daily_risk_free_rate
    )

    volatility = excess_returns.std()

    if volatility == 0:
        return np.nan

    return (
        excess_returns.mean()
        / volatility
        * np.sqrt(periods_per_year)
    )


def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal.

        Example:
        -0.25 = -25%
    """

    returns = returns.dropna()

    if returns.empty:
        return np.nan

    wealth_index = (
        1 + returns
    ).cumprod()

    running_peak = (
        wealth_index.cummax()
    )

    drawdown = (
        wealth_index
        / running_peak
        - 1
    )

    return drawdown.min()


def calculate_drawdown_series(
    returns
):
    """
    Calculate the complete drawdown
    series over time.
    """

    returns = returns.dropna()

    if returns.empty:
        return pd.Series(
            dtype=float
        )

    wealth_index = (
        1 + returns
    ).cumprod()

    running_peak = (
        wealth_index.cummax()
    )

    return (
        wealth_index
        / running_peak
        - 1
    )


def calculate_win_rate(returns):
    """
    Calculate percentage of positive-return
    periods.
    """

    returns = returns.dropna()

    if returns.empty:
        return np.nan

    winning_periods = (
        (returns > 0).sum()
    )

    return (
        winning_periods
        / len(returns)
    )


def calculate_average_return(
    returns
):
    """
    Calculate average periodic return.
    """

    returns = returns.dropna()

    if returns.empty:
        return np.nan

    return returns.mean()


def calculate_metrics(
    returns,
    risk_free_rate=0.02
):
    """
    Calculate a complete set of
    strategy performance metrics.

    Returns
    -------
    dict
    """

    returns = returns.dropna()

    return {
        "Total Return":
            calculate_total_return(
                returns
            ),

        "Annualized Return":
            calculate_annualized_return(
                returns
            ),

        "Volatility":
            calculate_volatility(
                returns
            ),

        "Sharpe Ratio":
            calculate_sharpe_ratio(
                returns,
                risk_free_rate
            ),

        "Maximum Drawdown":
            calculate_max_drawdown(
                returns
            ),

        "Win Rate":
            calculate_win_rate(
                returns
            ),

        "Average Daily Return":
            calculate_average_return(
                returns
            ),
    }


def print_metrics(
    metrics
):
    """
    Print metrics in a readable format.
    """

    print("=" * 50)
    print("BACKTEST PERFORMANCE")
    print("=" * 50)

    print(
        f"Total Return: "
        f"{metrics['Total Return'] * 100:.2f}%"
    )

    print(
        f"Annualized Return: "
        f"{metrics['Annualized Return'] * 100:.2f}%"
    )

    print(
        f"Volatility: "
        f"{metrics['Volatility'] * 100:.2f}%"
    )

    print(
        f"Sharpe Ratio: "
        f"{metrics['Sharpe Ratio']:.2f}"
    )

    print(
        f"Maximum Drawdown: "
        f"{metrics['Maximum Drawdown'] * 100:.2f}%"
    )

    print(
        f"Win Rate: "
        f"{metrics['Win Rate'] * 100:.2f}%"
    )

    print(
        f"Average Daily Return: "
        f"{metrics['Average Daily Return'] * 100:.4f}%"
    )

    print("=" * 50)