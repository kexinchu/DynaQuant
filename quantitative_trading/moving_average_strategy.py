import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime

# 定义均线策略
class MovingAverageStrategy(bt.Strategy):
    params = (("short_window", 50), ("long_window", 200))

    def __init__(self):
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data, period=self.params.short_window)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data, period=self.params.long_window)

    def next(self):
        if not self.position:  # 目前没有仓位
            if self.sma_short[0] > self.sma_long[0]:  # 均线金叉
                self.buy()  # 买入
        else:
            if self.sma_short[0] < self.sma_long[0]:  # 均线死叉
                self.sell()  # 卖出

# 获取数据
def get_stock_data(ticker, start="2020-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    return df

# 运行回测
def run_backtest(stock_ticker):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MovingAverageStrategy)

    data = bt.feeds.PandasData(dataname=get_stock_data(stock_ticker))
    cerebro.adddata(data)

    cerebro.broker.set_cash(100000)  # 设置初始资金
    cerebro.broker.setcommission(commission=0.001)  # 佣金设定

    print(f"初始资金: {cerebro.broker.getvalue()}")

    cerebro.run()

    print(f"最终资金: {cerebro.broker.getvalue()}")

    cerebro.plot()

# 选择美股标的
stocks = ["NVDA", "GOOG"]
for stock in stocks:
    print(f"回测 {stock}")
    run_backtest(stock)
