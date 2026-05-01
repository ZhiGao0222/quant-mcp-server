# Quant MCP Server

MCP tools for portfolio analytics, risk metrics, and quantitative finance research.

This project lets AI agents analyze local CSV price datasets through MCP tools.

## Features

- List local CSV datasets
- Summarize asset price data
- Calculate simple and log returns
- Compute annualized return, volatility, Sharpe ratio, and max drawdown
- Run a simple equal-weight portfolio backtest
- Suggest inverse-volatility portfolio weights

## Quickstart

Clone the repo:

    git clone https://github.com/YOUR_USERNAME/quant-mcp-server.git
    cd quant-mcp-server

Install dependencies:

    uv sync

Run MCP Inspector:

    uv run mcp dev src/quant_mcp_server/server.py --with-editable .

## Example Dataset

Put CSV files inside the data folder.

Example format:

    date,SPY,QQQ,TLT,GLD
    2024-01-02,470.0,400.0,98.0,190.0
    2024-01-03,468.5,397.2,98.8,191.1

## Available Tools

### list_datasets

Lists available CSV files in data/.

### summarize_prices

Summarizes rows, assets, date range, and latest prices.

### calculate_returns

Calculates simple or log returns.

### compute_risk_metrics

Computes annualized return, volatility, Sharpe ratio, and max drawdown.

### backtest_equal_weight_portfolio

Runs a simple equal-weight portfolio backtest.

### suggest_inverse_vol_weights

Suggests inverse-volatility portfolio weights.

## Example Prompt

Analyze sample_prices.csv. First summarize the dataset, then compute risk metrics, then run an equal-weight portfolio backtest.

## Disclaimer

This project is for education and research only. It is not financial advice and does not execute trades.
