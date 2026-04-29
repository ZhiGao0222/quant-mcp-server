from quant_mcp_server.server import calc_portfolio_return, calc_simple_stats, calc_target_leverage


def test_portfolio_return():
    out = calc_portfolio_return([0.5, 0.5], [0.1, 0.2])
    assert round(out["portfolio_return"], 6) == 0.15


def test_simple_stats():
    out = calc_simple_stats([1, 2, 3])
    assert out["mean"] == 2.0
    assert out["count"] == 3.0


def test_target_leverage():
    out = calc_target_leverage(1000, 3, 50)
    assert out["dollar_exposure"] == 3000
    assert out["shares"] == 60
    assert out["borrow_amount"] == 2000
