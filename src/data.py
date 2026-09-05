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
        Yahoo Finance period.

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

    # Handle yfinance MultiIndex columns
    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = (
            data.columns
            .get_level_values(0)
        )

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

    data = data.sort_index()

    # Make prices numeric
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

    # Make volume numeric
    if "Volume" in data.columns:

        data["Volume"] = pd.to_numeric(
            data["Volume"],
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
    Download only closing prices.

    Returns
    -------
    pandas.Series
    """

    data = get_stock_data(
        ticker,
        period
    )

    prices = data["Close"]

    # Safety check for yfinance
    if isinstance(
        prices,
        pd.DataFrame
    ):

        prices = prices.iloc[:, 0]

    prices.name = ticker.upper().strip()

    return prices


def get_multiple_stocks(
    tickers,
    period="5y"
):
    """
    Download historical data for
    multiple stocks.

    Parameters
    ----------
    tickers : list
        Example:
        ["AAPL", "MSFT", "NVDA"]

    period : str
        Yahoo Finance period.

    Returns
    -------
    dict
        Dictionary:

        {
            "AAPL": DataFrame,
            "MSFT": DataFrame,
            "NVDA": DataFrame
        }
    """

    stock_data = {}

    for ticker in tickers:

        ticker = ticker.upper().strip()

        if not ticker:
            continue

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