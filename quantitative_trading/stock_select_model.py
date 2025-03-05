# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# 加载数据
def load_data(ticker):
    df = pd.read_csv(f'{data_path}{ticker}.csv', index_col=0, parse_dates=True)
    return df

# 特征工程
def feature_engineering(df):
    df['SMA_10'] = SMAIndicator(df['Close'], window=10).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
    df = df.dropna()
    return df

# 训练模型
def train_model(df):
    features = ['SMA_10', 'SMA_50', 'RSI']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'模型准确率: {accuracy:.2f}')
    return model

# 加载并处理所有股票数据
all_data = []
for ticker in nasdaq_100_tickers:
    df = load_data(ticker)
    df = feature_engineering(df)
    df['Ticker'] = ticker
    all_data.append(df)

# 合并所有数据
all_data = pd.concat(all_data)

# 训练模型
model = train_model(all_data)
