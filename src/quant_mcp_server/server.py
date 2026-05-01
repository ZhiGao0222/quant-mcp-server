from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("quant-mcp-server")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRADING_DAYS = 252


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _safe_data_path(filename: str) -> Path:
    """
    Only allow CSV files inside the local data/ directory.

    This prevents the MCP tool from reading arbitrary files on your machine.
    """
    _ensure_data_dir()

    candidate = (DATA_DIR / filename).resolve()
    base = DATA_DIR.resolve()

    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("Only files inside the data/ directory are allowed.") from exc

    if candidate.suffix.lower() != ".csv":
        raise ValueError("Only .csv files are supported.")

    if not candidate.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    return candidate


def _load_prices(filename: str) -> pd.DataFrame:
    path = _safe_data_path(filename)
    df = pd.read_csv(path)

    date_col = next(
        (col for col in df.columns if col.lower() in {"date", "datetime", "timestamp"}),
        None,
    )

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    prices = df.select_dtypes(include=["number"]).copy()

    if prices.empty:
        raise ValueError("CSV must contain at least one numeric price column.")

    prices = prices.dropna(how="all")

    if len(prices) < 2:
        raise ValueError("Need at least two rows of prices.")

    return prices


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.reset_index()

    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)

    return out.to_dict(orient="records")


@mcp.tool()
def list_datasets() -> list[str]:
    """
    List available CSV datasets in the local data/ directory.
    """
    _ensure_data_dir()
    return sorted(path.name for path in DATA_DIR.glob("*.csv"))


@mcp.tool()
def summarize_prices(filename: str) -> dict[str, Any]:
    """
    Summarize a CSV price dataset.

    Args:
        filename: CSV file name inside the data/ directory.
    """
    prices = _load_prices(filename)

    start = str(prices.index.min()) if len(prices.index) else None
    end = str(prices.index.max()) if len(prices.index) else None

    last_prices = {
        asset: float(value)
        for asset, value in prices.tail(1).iloc[0].round(4).items()
    }

    return {
        "filename": filename,
        "rows": int(len(prices)),
        "assets": list(prices.columns),
        "start": start,
        "end": end,
        "last_prices": last_prices,
    }


@mcp.tool()
def calculate_returns(
    filename: str,
    method: str = "simple",
    last_n: int = 5,
) -> dict[str, Any]:
    """
    Calculate recent asset returns from a local price CSV.

    Args:
        filename: CSV file name inside the data/ directory.
        method: "simple" for percentage returns, or "log" for log returns.
        last_n: Number of latest return rows to return.
    """
    prices = _load_prices(filename)

    if method not in {"simple", "log"}:
        raise ValueError('method must be either "simple" or "log".')

    if method == "simple":
        returns = prices.pct_change().dropna(how="all")
    else:
        returns = np.log(prices / prices.shift(1)).dropna(how="all")

    last_n = max(1, min(last_n, 50))
    returns = returns.tail(last_n).round(6)

    return {
        "filename": filename,
        "method": method,
        "last_n": last_n,
        "returns": _to_records(returns),
    }


@mcp.tool()
def compute_risk_metrics(
    filename: str,
    annualization: int = TRADING_DAYS,
) -> dict[str, Any]:
    """
    Compute annualized return, volatility, Sharpe ratio, and max drawdown per asset.

    Args:
        filename: CSV file name inside the data/ directory.
        annualization: Annualization factor. Use 252 for daily data.
    """
    prices = _load_prices(filename)
    returns = prices.pct_change().dropna(how="all")

    metrics: dict[str, Any] = {}

    for asset in returns.columns:
        r = returns[asset].dropna()

        if r.empty:
            continue

        annual_return = float(r.mean() * annualization)
        annual_volatility = float(r.std(ddof=1) * np.sqrt(annualization))
        sharpe = annual_return / annual_volatility if annual_volatility != 0 else None
        max_dd = _max_drawdown(r)

        metrics[asset] = {
            "annual_return": round(annual_return, 6),
            "annual_volatility": round(annual_volatility, 6),
            "sharpe_ratio": round(sharpe, 6) if sharpe is not None else None,
            "max_drawdown": round(max_dd, 6),
        }

    return {
        "filename": filename,
        "annualization": annualization,
        "metrics": metrics,
    }


@mcp.tool()
def backtest_equal_weight_portfolio(
    filename: str,
    initial_cash: float = 10000.0,
    annualization: int = TRADING_DAYS,
) -> dict[str, Any]:
    """
    Run a simple daily equal-weight portfolio backtest.

    Args:
        filename: CSV file name inside the data/ directory.
        initial_cash: Starting portfolio value.
        annualization: Annualization factor. Use 252 for daily data.
    """
    prices = _load_prices(filename).dropna()
    returns = prices.pct_change().dropna(how="all")

    portfolio_returns = returns.mean(axis=1)
    equity = initial_cash * (1 + portfolio_returns).cumprod()

    total_return = float(equity.iloc[-1] / initial_cash - 1)
    annual_return = float(portfolio_returns.mean() * annualization)
    annual_volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(annualization))
    sharpe = annual_return / annual_volatility if annual_volatility != 0 else None
    max_dd = _max_drawdown(portfolio_returns)

    equity_curve = pd.DataFrame({"equity": equity.round(2)}).tail(10)

    return {
        "filename": filename,
        "initial_cash": initial_cash,
        "ending_value": round(float(equity.iloc[-1]), 2),
        "total_return": round(total_return, 6),
        "annual_return": round(annual_return, 6),
        "annual_volatility": round(annual_volatility, 6),
        "sharpe_ratio": round(sharpe, 6) if sharpe is not None else None,
        "max_drawdown": round(max_dd, 6),
        "assets": list(prices.columns),
        "weights": {asset: round(1 / len(prices.columns), 6) for asset in prices.columns},
        "recent_equity_curve": _to_records(equity_curve),
    }


@mcp.tool()
def suggest_inverse_vol_weights(
    filename: str,
    lookback_days: int = 252,
) -> dict[str, Any]:
    """
    Suggest inverse-volatility portfolio weights.

    Assets with lower recent volatility receive higher weights.

    Args:
        filename: CSV file name inside the data/ directory.
        lookback_days: Number of recent return rows used to estimate volatility.
    """
    prices = _load_prices(filename)
    returns = prices.pct_change().dropna(how="all")

    lookback_days = max(2, min(lookback_days, len(returns)))
    window = returns.tail(lookback_days)

    vol = window.std(ddof=1).replace(0, np.nan)
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()

    clean_weights = {
        asset: round(float(weight), 6)
        for asset, weight in weights.dropna().items()
    }

    return {
        "filename": filename,
        "lookback_days": lookback_days,
        "method": "inverse_volatility",
        "weights": clean_weights,
    }


@mcp.prompt()
def quant_research_prompt(filename: str) -> str:
    """
    Create a reusable prompt for analyzing a local price dataset.
    """
    return (
        f"Analyze the local price dataset `{filename}`. "
        "First summarize the assets and date range. "
        "Then compute risk metrics, run an equal-weight backtest, "
        "and explain the main risks in simple language. "
        "Do not give financial advice."
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
