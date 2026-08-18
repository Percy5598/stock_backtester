from src.data import get_stock_data
from src.comparison import (
    compare_strategies,
    print_comparison,
)


TICKER = "AAPL"
PERIOD = "5y"
INITIAL_CAPITAL = 10_000


data = get_stock_data(
    TICKER,
    period=PERIOD
)

prices = data["Close"]

if hasattr(prices, "columns"):
    prices = prices.iloc[:, 0]


comparison = compare_strategies(
    prices,
    initial_capital=INITIAL_CAPITAL
)


print_comparison(
    comparison
)