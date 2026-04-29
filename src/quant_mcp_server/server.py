from mcp.server.fastmcp import FastMCP
import numpy as np
import pandas as pd

mcp = FastMCP("Quant MCP Server")


def calc_portfolio_return(weights: list[float], returns: list[float]) -> dict[str, float]:
    if len(weights) != len(returns):
        raise ValueError("weights and returns need same length")

    w = np.array(weights, dtype=float)
    r = np.array(returns, dtype=float)

    total = float(w.sum())
    if total == 0:
        raise ValueError("weights cannot add to 0")

    w = w / total
    value = float(np.dot(w, r))

    return {"portfolio_return": value}


def calc_simple_stats(values: list[float]) -> dict[str, float]:
    if len(values) == 0:
        raise ValueError("values cannot be empty")

    s = pd.Series(values, dtype="float64")

    std_value = 0.0
    if len(values) > 1:
        std_value = float(s.std(ddof=1))

    return {
        "count": float(s.count()),
        "mean": float(s.mean()),
        "std": std_value,
        "min": float(s.min()),
        "max": float(s.max()),
    }


def calc_target_leverage(
    equity: float,
    target_leverage: float,
    asset_price: float,
) -> dict[str, float]:
    if equity <= 0:
        raise ValueError("equity must be positive")
    if asset_price <= 0:
        raise ValueError("asset_price must be positive")

    exposure = equity * target_leverage
    shares = exposure / asset_price
    borrow = max(exposure - equity, 0.0)

    return {
        "dollar_exposure": exposure,
        "shares": shares,
        "borrow_amount": borrow,
    }


@mcp.tool()
def portfolio_return(weights: list[float], returns: list[float]) -> dict[str, float]:
    """Calculate weighted portfolio return."""
    return calc_portfolio_return(weights, returns)


@mcp.tool()
def simple_stats(values: list[float]) -> dict[str, float]:
    """Return basic statistics for a list of numbers."""
    return calc_simple_stats(values)


@mcp.tool()
def target_leverage_position(
    equity: float,
    target_leverage: float,
    asset_price: float,
) -> dict[str, float]:
    """Estimate position size for a target leverage."""
    return calc_target_leverage(equity, target_leverage, asset_price)


@mcp.tool()
def hello(message: str = "hi") -> str:
    """Check that the MCP server is working."""
    return f"quant mcp server is working: {message}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
