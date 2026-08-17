"""
离线版的股票收益数据生成器, 替代原书里 pandas_datareader.get_data_yahoo(...)
对雅虎财经(Yahoo Finance)的实时抓取。

原书代码需要联网访问雅虎财经的历史行情接口(而且这个接口本身近几年也
经常变动、失效),在没有网络访问权限的环境下(比如本仓库构建时所在的
沙箱环境)必然会失败。

本模块用固定随机种子, 生成一份*统计结构上*与真实股票日收益率相似的合成
数据: 小幅度的日波动率、股票之间存在一定的相关性(通过协方差矩阵控制)、
不同标的的波动率高低有差异(模拟"TSLA 波动更大、AAPL 相对稳健"这种真实
世界里常见的现象)。

**重要说明**: 下面生成的收益率数值是完全合成的, 并不是 AAPL/GOOG/TSLA/
AMZN 的真实历史日收益率, 只是为了让本章"贝叶斯建模股票收益协方差矩阵"
这个例子能够离线端到端跑通, 同时统计上的相关结构依然能支撑后续的
Wishart 协方差矩阵推断这一教学目的。如果你有网络访问权限, 可以直接换回
原书使用 pandas_datareader (或者其他行情数据源) 拉取真实历史数据。
"""
import numpy as np
import pandas as pd


def load_stock_returns(stocks, startdate="2012-09-01", enddate="2015-04-27", seed=7):
    """返回一个 DataFrame, index 是交易日日期, 列是 stocks 里的股票代码,
    值是(合成的)日收益率。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=startdate, end=enddate)

    n = len(stocks)
    # 合成的日波动率(标准差), 大致模仿真实世界里不同股票的波动差异:
    # TSLA 波动明显更大, AAPL/GOOG/AMZN 相对温和。
    vol_map = {"AAPL": 0.017, "GOOG": 0.016, "TSLA": 0.035, "AMZN": 0.020}
    vols = np.array([vol_map.get(s, 0.02) for s in stocks])

    # 合成一个温和正相关的相关矩阵(同属科技板块, 收益率往往有一定的
    # 共同波动), 对角线为 1。
    base_corr = 0.25
    corr = np.full((n, n), base_corr)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr

    means = np.array([0.0008, -0.0002, 0.0015, 0.0010])[:n] if n <= 4 else rng.normal(0, 0.001, n)

    returns = rng.multivariate_normal(mean=means, cov=cov, size=len(dates))
    return pd.DataFrame(returns, index=dates, columns=stocks)
