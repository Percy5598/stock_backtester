import matplotlib.pyplot as plt


def plot_equity_curves(
    results,
    title="Strategy Performance",
    save_path="equity_curves.png",
):
    """
    Plot and save portfolio equity curves.

    Parameters
    ----------
    results : dict
        Strategy name -> backtest DataFrame.

    title : str
        Chart title.

    save_path : str
        Output image path.
    """

    if not results:
        raise ValueError("Results cannot be empty.")

    fig, ax = plt.subplots(figsize=(12, 6))

    for strategy_name, data in results.items():

        if "Portfolio Value" not in data.columns:
            raise ValueError(
                f"'Portfolio Value' missing for {strategy_name}"
            )

        ax.plot(
            data.index,
            data["Portfolio Value"],
            label=strategy_name,
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (€)")

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Equity curve saved to: {save_path}"
    )