# -*- coding:utf-8 -*-

import yfinance as yf

# 获取纳斯达克100指数成分股列表
nasdaq_100_tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'FB', 'TSLA', 'NVDA', 'PYPL', 'ADBE',
    'CMCSA', 'NFLX', 'INTC', 'PEP', 'CSCO', 'AVGO', 'COST', 'AMGN', 'CHTR', 'TXN',
    'QCOM', 'GILD', 'SBUX', 'ISRG', 'MDLZ', 'BKNG', 'FISV', 'ATVI', 'ADP', 'INTU',
    'AMD', 'MU', 'AMAT', 'ZM', 'LRCX', 'REGN', 'CSX', 'VRTX', 'ADSK', 'BIIB',
    'KHC', 'MELI', 'JD', 'ILMN', 'WBA', 'ROST', 'IDXX', 'MNST', 'ORLY', 'CTSH',
    'NXPI', 'LULU', 'EA', 'EXC', 'BIDU', 'KDP', 'XEL', 'ALGN', 'MAR', 'SNPS',
    'CTAS', 'PCAR', 'VRSK', 'PAYX', 'ASML', 'EBAY', 'TEAM', 'SGEN', 'ANSS', 'SWKS',
    'WDAY', 'FTNT', 'SPLK', 'DLTR', 'OKTA', 'MRVL', 'CHKP', 'DOCU', 'CPRT', 'NTES',
    'VRSN', 'TCOM', 'XLNX', 'PDD', 'MCHP', 'CDW', 'FAST', 'WDC', 'MXIM', 'DLTR',
    'ULTA', 'TTWO', 'INCY', 'LBTYK', 'LBTYA', 'LBTYB', 'SIRI', 'FOX', 'FOXA'
]

# 定义数据保存路径
data_path = 'nasdaq_100_data/'

# 创建文件夹
import os
if not os.path.exists(data_path):
    os.makedirs(data_path)

# 下载并保存数据
start_date = '2023-01-31'
end_date = '2025-01-31'

for ticker in nasdaq_100_tickers:
    print(f'正在下载 {ticker} 的数据...')
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(f'{data_path}{ticker}.csv')
    print(f'{ticker} 的数据已保存。\n')
