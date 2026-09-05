import streamlit as st
import pandas as pd

from src.data import get_close_prices
from src.comparison import (
    compare_multiple_stocks,
    format_comparison,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Backtester",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 Stock Backtester")

st.write(
    """
    Compare different trading strategies across multiple stocks
    using historical market data.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Backtest Settings")


# ------------------------------------------------------------
# Stock selection
# ------------------------------------------------------------

st.sidebar.subheader("Stocks")

stock_input = st.sidebar.text_input(
    "Enter stock tickers",
    value="AAPL, MSFT, NVDA",
    help="Enter multiple tickers separated by commas."
)

tickers = [
    ticker.strip().upper()
    for ticker in stock_input.split(",")
    if ticker.strip()
]


# ------------------------------------------------------------
# Historical period
# ------------------------------------------------------------

period = st.sidebar.selectbox(
    "Historical period",
    options=[
        "1y",
        "2y",
        "3y",
        "5y",
        "10y",
        "max",
    ],
    index=3,
)


# ------------------------------------------------------------
# Initial capital
# ------------------------------------------------------------

initial_capital = st.sidebar.number_input(
    "Initial capital",
    min_value=100,
    max_value=10_000_000,
    value=10_000,
    step=1_000,
)


# ============================================================
# STRATEGY SELECTION
# ============================================================

st.sidebar.header("Strategies")

selected_strategies = []


# ------------------------------------------------------------
# Buy & Hold
# ------------------------------------------------------------

if st.sidebar.checkbox(
    "Buy & Hold",
    value=True,
):
    selected_strategies.append(
        "buy_and_hold"
    )


# ------------------------------------------------------------
# Moving Average
# ------------------------------------------------------------

moving_average_selected = st.sidebar.checkbox(
    "Moving Average",
    value=True,
)

moving_average_window = 20

if moving_average_selected:

    selected_strategies.append(
        "moving_average"
    )

    moving_average_window = (
        st.sidebar.slider(
            "MA window",
            min_value=5,
            max_value=200,
            value=20,
            step=5,
        )
    )


# ------------------------------------------------------------
# Moving Average Crossover
# ------------------------------------------------------------

crossover_selected = st.sidebar.checkbox(
    "MA Crossover",
    value=True,
)

short_window = 20
long_window = 50

if crossover_selected:

    selected_strategies.append(
        "moving_average_crossover"
    )

    short_window = st.sidebar.slider(
        "Short MA window",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
    )

    long_window = st.sidebar.slider(
        "Long MA window",
        min_value=20,
        max_value=300,
        value=50,
        step=5,
    )


# ------------------------------------------------------------
# Momentum
# ------------------------------------------------------------

momentum_selected = st.sidebar.checkbox(
    "Momentum",
    value=True,
)

momentum_lookback = 20

if momentum_selected:

    selected_strategies.append(
        "momentum"
    )

    momentum_lookback = (
        st.sidebar.slider(
            "Momentum lookback",
            min_value=5,
            max_value=200,
            value=20,
            step=5,
        )
    )


# ============================================================
# STRATEGY PARAMETERS
# ============================================================

strategy_parameters = {}


if moving_average_selected:

    strategy_parameters[
        "moving_average"
    ] = {
        "window": moving_average_window
    }


if crossover_selected:

    strategy_parameters[
        "moving_average_crossover"
    ] = {
        "short_window": short_window,
        "long_window": long_window,
    }


if momentum_selected:

    strategy_parameters[
        "momentum"
    ] = {
        "lookback": momentum_lookback
    }


# ============================================================
# RUN BACKTEST
# ============================================================

run_backtest_button = st.sidebar.button(
    "🚀 Run Backtest",
    type="primary",
)


if run_backtest_button:

    if not tickers:

        st.error(
            "Please enter at least one stock ticker."
        )

        st.stop()


    if not selected_strategies:

        st.error(
            "Please select at least one strategy."
        )

        st.stop()


    if (
        crossover_selected
        and short_window >= long_window
    ):

        st.error(
            "Short MA window must be smaller "
            "than Long MA window."
        )

        st.stop()


    # --------------------------------------------------------
    # Download stock data
    # --------------------------------------------------------

    stock_prices = {}

    progress = st.progress(0)

    status = st.empty()

    for index, ticker in enumerate(tickers):

        status.write(
            f"Downloading {ticker}..."
        )

        try:

            prices = get_close_prices(
                ticker,
                period=period,
            )

            stock_prices[ticker] = prices

        except Exception as error:

            st.warning(
                f"Could not download {ticker}: {error}"
            )

        progress.progress(
            (index + 1) / len(tickers)
        )


    status.empty()
    progress.empty()


    if not stock_prices:

        st.error(
            "No stock data could be downloaded."
        )

        st.stop()


    # --------------------------------------------------------
    # Run comparison
    # --------------------------------------------------------

    with st.spinner(
        "Running backtests..."
    ):

        comparison, backtest_results = (
            compare_multiple_stocks(
                stock_prices,
                initial_capital=initial_capital,
                strategies=selected_strategies,
                strategy_parameters=strategy_parameters,
            )
        )


    # ========================================================
    # RESULTS
    # ========================================================

    st.header("Backtest Results")


    # --------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------

    st.subheader(
        "Strategy Comparison"
    )

    formatted_comparison = (
        format_comparison(
            comparison
        )
    )

    st.dataframe(
        formatted_comparison,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # EQUITY CURVES
    # ========================================================

    st.subheader(
        "Portfolio Performance"
    )

    for ticker in stock_prices:

        st.write(
            f"### {ticker}"
        )

        ticker_results = (
            backtest_results[ticker]
        )

        chart_data = pd.DataFrame()

        for strategy_name, results in (
            ticker_results.items()
        ):

            chart_data[strategy_name] = (
                results["Portfolio Value"]
            )

        st.line_chart(
            chart_data,
            use_container_width=True,
        )


    # ========================================================
    # DRAWDOWN
    # ========================================================

    st.subheader(
        "Drawdown"
    )

    for ticker in stock_prices:

        st.write(
            f"### {ticker}"
        )

        ticker_results = (
            backtest_results[ticker]
        )

        drawdown_data = pd.DataFrame()

        for strategy_name, results in (
            ticker_results.items()
        ):

            portfolio = (
                results["Portfolio Value"]
            )

            running_max = (
                portfolio.cummax()
            )

            drawdown = (
                portfolio / running_max - 1
            )

            drawdown_data[strategy_name] = (
                drawdown
            )

        st.line_chart(
            drawdown_data,
            use_container_width=True,
        )


else:

    # ========================================================
    # LANDING SCREEN
    # ========================================================

    st.info(
        """
        Configure your backtest in the sidebar.

        **Example**

        Stocks:
        `AAPL, MSFT, NVDA`

        Period:
        `5 years`

        Initial capital:
        `€10,000`

        Then click **Run Backtest**.
        """
    )