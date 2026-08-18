import pandas as pd
import yfinance as yf


def get_stock_data(
    ticker,
    period="5y"
):
    """
    Download historical stock data.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. "AAPL".

    period : str
        Yahoo Finance period, e.g.
        "1y", "2y", "5y", "max".

    Returns
    -------
    pandas.DataFrame
        Historical OHLCV data.
    """

    ticker = ticker.upper().strip()

    if not ticker:
        raise ValueError(
            "Ticker cannot be empty."
        )

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No data found for {ticker}."
        )

    # --------------------------------------------------
    # Handle yfinance MultiIndex columns
    # --------------------------------------------------

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        # For a single ticker, remove the
        # ticker level from the columns.
        data.columns = (
            data.columns
            .get_level_values(0)
        )

    # --------------------------------------------------
    # Keep only the columns needed by the
    # backtesting project.
    # --------------------------------------------------

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    available_columns = [
        column
        for column in required_columns
        if column in data.columns
    ]

    data = data[
        available_columns
    ].copy()

    # --------------------------------------------------
    # Clean the data
    # --------------------------------------------------

    data = data.sort_index()

    data = data.dropna(
        subset=["Close"]
    )

    # Make sure Volume is numeric
    if "Volume" in data.columns:

        data["Volume"] = pd.to_numeric(
            data["Volume"],
            errors="coerce"
        )

    # Make sure prices are numeric
    price_columns = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    for column in price_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    data = data.dropna(
        subset=["Close"]
    )

    return data


def get_close_prices(
    ticker,
    period="5y"
):
    """
    Download only the closing prices
    for a stock.

    Returns
    -------
    pandas.Series
    """

    data = get_stock_data(
        ticker,
        period
    )

    return data["Close"]


def get_multiple_stocks(
    tickers,
    period="5y"
):
    """
    Download data for multiple stocks.

    Parameters
    ----------
    tickers : list
        Example:
        ["AAPL", "MSFT", "NVDA"]

    Returns
    -------
    dict
        Dictionary containing DataFrames.
    """

    stock_data = {}

    for ticker in tickers:

        try:

            stock_data[ticker] = (
                get_stock_data(
                    ticker,
                    period
                )
            )

        except Exception as error:

            print(
                f"Could not download "
                f"{ticker}: {error}"
            )

    return stock_data