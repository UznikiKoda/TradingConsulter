from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib.pylab import date2num
from ta import add_all_ta_features
import ta
from ta.trend import MACD
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands
from ta.trend import PSARIndicator
from ta.momentum import RSIIndicator
from ta.momentum import StochasticOscillator
import requests
import apimoex
import datetime
from dateutil.parser import parse
from logger_TC import Logger
import os
from dotenv import load_dotenv
import psycopg2



def get_tickers():
    url = 'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json?'
    data = requests.get(url).json()
    tickers = [row[0] for row in data['securities']['data']]
    return tickers


class Indicators:
    def __init__(self, logger, ticker, date, interval='day', days=365, balance=10 ** 5,
                 comission=False):
        if interval == 'day':
            self.interval = 24
        elif interval == 'minute':
            self.interval = 1
        self.days = days
        start = parse(date)
        end = start + datetime.timedelta(days=self.days)

        with requests.Session() as session:
            data = apimoex.get_market_candles(session, security=ticker, interval=self.interval, start=start, end=end)

        self.comission = comission
        self.df = pd.DataFrame(data)
        self.df.columns = ['<DATE>', '<OPEN>', '<CLOSE>', '<HIGH>', '<LOW>', '<VALUE>', '<VOL>']
        self.balance = balance
        self.df["<VOL>"] = pd.to_numeric(self.df["<VOL>"], errors='coerce')
        self.df["<VOL>"] = self.df["<VOL>"].fillna(0)
        self.df = add_all_ta_features(self.df, open="<OPEN>", high="<HIGH>", low="<LOW>", close="<CLOSE>",
                                      volume="<VOL>", fillna=True)
        self.df['Buy_Signal'] = False
        self.df['Sell_Signal'] = False
        logger.log_start()
        logger.log_settings(ticker, date, balance, interval, days)
        logger.log_postgres(ticker, date, balance , interval, days)


    def macd_aroon(self, window_fast=11, window_slow=27, window_sign=9, lb=25, plot=True):
        _MACD = MACD(self.df["<CLOSE>"], window_fast=window_fast, window_slow=window_slow, window_sign=window_sign)
        self.df["MACD"] = _MACD.macd()
        self.df["MACD_signal"] = _MACD.macd_signal()
        self.df["MACD_gist"] = _MACD.macd_diff()
        self.df['AROONu'] = 100 * self.df['<HIGH>'].rolling(lb + 1).apply(lambda x: x.argmax()) / lb
        self.df['AROONd'] = 100 * self.df['<LOW>'].rolling(lb + 1).apply(lambda x: x.argmin()) / lb

        macd_aroon_df = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            action = self.check_conditions_macd_aroon(i, balance, count) if i != 0 else 'nothing'
            match action:
                case 'buy':
                    num = balance // self.df['<CLOSE>'][i]
                    count += balance // self.df['<CLOSE>'][i]
                    price = self.df['<CLOSE>'][i] * num
                    balance = balance - price
                    k += price * 0.01
                    day_status.append('buy')
                    self.df.at[i, 'Buy_Signal'] = True

                case 'sell':
                    balance += count * self.df['<CLOSE>'][i]
                    k += count * self.df['<CLOSE>'][i] * 0.01
                    count = 0
                    price = self.df['<CLOSE>'][i]
                    day_status.append('sell')
                    self.df.at[i, 'Sell_Signal'] = True

                case 'nothing':
                    day_status.append('nothing')

            if i == (len(self.df) - 1):
                if self.comission:
                    print(f'Комиссия за всё время(MACD + AROON): {k}')
                    balance -= k

                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]

            count_df.append(count)
            balance_df.append(balance)
            price_df.append(price)

        macd_aroon_df["NumOfShares"] = count_df
        macd_aroon_df["MoneySpent"] = price_df
        macd_aroon_df["Balance"] = balance_df
        macd_aroon_df["DayStatus"] = day_status

        if plot:
            subplots_data = [
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'black'}
                    ],
                    'title': 'Цены закрытия акций',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#f0f0f0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['MACD_signal'], 'label': 'Signal line', 'color': 'blue'},
                        {'x': self.df['<DATE>'], 'y': self.df['MACD'], 'label': 'MACD line', 'color': 'orange'}
                    ],
                    'bar': [
                        {'x': self.df['<DATE>'], 'y': self.df['MACD_gist'], 'label': 'MACD Histogram',
                         'color': self.df['MACD_gist'].apply(lambda x: '#ef5350' if x < 0 else '#26a69a')}
                    ],
                    'title': 'Индикатор MACD',
                    'xlabel': 'Дата',
                    'ylabel': 'MACD',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['AROONu'], 'label': 'AROON Up Channel', 'color': 'blue'},
                        {'x': self.df['<DATE>'], 'y': self.df['AROONd'], 'label': 'AROON Down Channel', 'color': 'orange'}
                    ],
                    'title': 'Индикатор AROON',
                    'xlabel': 'Дата',
                    'ylabel': 'Stochastic',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'blue'}
                    ],
                    'scatter': [
                        {'x': self.df[self.df['Buy_Signal']].index, 'y': self.df[self.df['Buy_Signal']]['<CLOSE>'],
                         'label': 'Покупка', 'color': 'green', 'size': 100},
                        {'x': self.df[self.df['Sell_Signal']].index, 'y': self.df[self.df['Sell_Signal']]['<CLOSE>'],
                         'label': 'Продажа', 'color': 'red', 'size': 100}
                    ],
                    'title': 'Торговые сигналы на основе MACD + AROON',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                }
            ]

            plot_data_with_subplots(subplots_data)

        return macd_aroon_df

    def atr_aroon(self, window=14, lb=25, plot=True):
        self.df['AROONu'] = 100 * self.df['<HIGH>'].rolling(lb + 1).apply(lambda x: x.argmax()) / lb
        self.df['AROONd'] = 100 * self.df['<LOW>'].rolling(lb + 1).apply(lambda x: x.argmin()) / lb
        high_low = self.df['<HIGH>'] - self.df['<LOW>']
        high_close = np.abs(self.df['<HIGH>'] - self.df['<CLOSE>'].shift())
        low_close = np.abs(self.df['<CLOSE>'].shift() - self.df['<LOW>'])
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.df['ATR'] = true_range.rolling(window).sum() / window
        self.df["SMAATR"] = SMAIndicator(close=self.df["ATR"], window=14).sma_indicator()

        atr_aroon_df = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            action = self.check_conditions_atr_aroon(i, balance, count) if i != 0 else 'nothing'
            match action:
                case 'buy':
                    num = balance // self.df['<CLOSE>'][i]
                    count += balance // self.df['<CLOSE>'][i]
                    price = self.df['<CLOSE>'][i] * num
                    balance = balance - price
                    k += price * 0.01
                    day_status.append('buy')
                    self.df.at[i, 'Buy_Signal'] = True

                case 'sell':
                    balance += count * self.df['<CLOSE>'][i]
                    k += count * self.df['<CLOSE>'][i] * 0.01
                    count = 0
                    price = self.df['<CLOSE>'][i]
                    day_status.append('sell')
                    self.df.at[i, 'Sell_Signal'] = True

                case 'nothing':
                    day_status.append('nothing')

            if i == (len(self.df) - 1):
                if self.comission:
                    print(f'Комиссия за всё время(ATR + AROON): {k}')
                    balance -= k

                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]

            count_df.append(count)
            balance_df.append(balance)
            price_df.append(price)

        atr_aroon_df["NumOfShares"] = count_df
        atr_aroon_df["MoneySpent"] = price_df
        atr_aroon_df["Balance"] = balance_df
        atr_aroon_df["DayStatus"] = day_status

        if plot:
            subplots_data = [
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'black'}
                    ],
                    'title': 'Цены закрытия акций',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#f0f0f0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['ATR'], 'label': 'ATR', 'color': 'blue'},
                        {'x': self.df['<DATE>'], 'y': self.df['SMAATR'], 'label': 'SMA for ATR', 'color': 'orange'}
                    ],
                    'title': 'Индикатор ATR',
                    'xlabel': 'Дата',
                    'ylabel': 'IVAR',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['AROONu'], 'label': 'AROON Up Channel', 'color': 'red'},
                        {'x': self.df['<DATE>'], 'y': self.df['AROONd'], 'label': 'AROON Down Channel', 'color': 'red'}
                    ],
                    'title': 'Индикатор AROON',
                    'xlabel': 'Дата',
                    'ylabel': 'IVAR',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'blue'}
                    ],
                    'scatter': [
                        {'x': self.df[self.df['Buy_Signal']].index, 'y': self.df[self.df['Buy_Signal']]['<CLOSE>'],
                         'label': 'Покупка', 'color': 'green', 'size': 100},
                        {'x': self.df[self.df['Sell_Signal']].index, 'y': self.df[self.df['Sell_Signal']]['<CLOSE>'],
                         'label': 'Продажа', 'color': 'red', 'size': 100}
                    ],
                    'title': 'Торговые сигналы на основе ATR + AROON',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                }
            ]

            plot_data_with_subplots(subplots_data)

        return atr_aroon_df

    def bollinger_bands(self, window=20, window_dev=2, plot=True):
        indicator_bb = BollingerBands(close=self.df["<CLOSE>"], window=window, window_dev=window_dev)
        self.df['bb_bbm'] = indicator_bb.bollinger_mavg()
        self.df['bb_bbh'] = indicator_bb.bollinger_hband()
        self.df['bb_bbl'] = indicator_bb.bollinger_lband()
        self.df["SMA"] = SMAIndicator(close=self.df["<CLOSE>"], window=2).sma_indicator()

        bb_df = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            action = self.check_conditions_bb(i, balance, count) if i != 0 else 'nothing'
            match action:
                case 'buy':
                    count += balance // self.df['<CLOSE>'][i]
                    price = self.df['<CLOSE>'][i] * count
                    balance = balance - price
                    k += price * 0.01
                    day_status.append('buy')
                    self.df.at[i, 'Buy_Signal'] = True

                case 'sell':
                    balance += count * self.df['<CLOSE>'][i]
                    k += count * self.df['<CLOSE>'][i] * 0.01
                    count = 0
                    price = self.df['<CLOSE>'][i]
                    day_status.append('sell')
                    self.df.at[i, 'Sell_Signal'] = True

                case 'nothing':
                    day_status.append('nothing')

            if i == (len(self.df) - 1):
                if self.comission:
                    print(f'Комиссия за всё время(Bolinger Bands): {k}')
                    balance -= k

                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]

            count_df.append(count)
            balance_df.append(balance)
            price_df.append(price)

        bb_df["NumOfShares"] = count_df
        bb_df["MoneySpent"] = price_df
        bb_df["Balance"] = balance_df
        bb_df["DayStatus"] = day_status

        if plot:
            subplots_data = [
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'black'},
                    ],
                    'title': 'Цены закрытия акций',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#f0f0f0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'black'},
                        {'x': self.df['<DATE>'], 'y': self.df['bb_bbm'], 'label': 'BB Middle', 'color': 'yellow'},
                        {'x': self.df['<DATE>'], 'y': self.df['bb_bbh'], 'label': 'BB High', 'color': 'green'},
                        {'x': self.df['<DATE>'], 'y': self.df['bb_bbl'], 'label': 'BB Low', 'color': 'red'},
                        {'x': self.df['<DATE>'], 'y': self.df['SMA'], 'label': 'SMA', 'color': 'orange'}
                    ],
                    'title': 'Индикатор Bollinger Bands вместе с SMA',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#f0f0f0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'blue'}
                    ],
                    'scatter': [
                        {'x': self.df[self.df['Buy_Signal']].index, 'y': self.df[self.df['Buy_Signal']]['<CLOSE>'],
                         'label': 'Покупка', 'color': 'green', 'size': 100},
                        {'x': self.df[self.df['Sell_Signal']].index, 'y': self.df[self.df['Sell_Signal']]['<CLOSE>'],
                         'label': 'Продажа', 'color': 'red', 'size': 100}
                    ],
                    'title': 'Торговые сигналы на основе Bollinger Bands вместе с SMA',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                }
            ]

            plot_data_with_subplots(subplots_data)

        return bb_df

    def psar(self, step=0.02, max_step=0.2, plot=True):
        psar = PSARIndicator(high=self.df["<HIGH>"], low=self.df["<LOW>"], close=self.df["<CLOSE>"], step=step,
                             max_step=max_step)
        self.df["PSARu"] = psar.psar_up()
        self.df["PSARd"] = psar.psar_down()

        psar_df = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            action = self.check_conditions_psar(i, balance, count) if i != 0 else 'nothing'
            match action:
                case 'buy':
                    count += balance // self.df['<CLOSE>'][i]
                    price = self.df['<CLOSE>'][i] * count
                    balance = balance - price
                    k += price * 0.01
                    day_status.append('buy')
                    self.df.at[i, 'Buy_Signal'] = True

                case 'sell':
                    balance += count * self.df['<CLOSE>'][i]
                    k += count * self.df['<CLOSE>'][i] * 0.01
                    count = 0
                    price = self.df['<CLOSE>'][i]
                    day_status.append('sell')
                    self.df.at[i, 'Sell_Signal'] = True

                case 'nothing':
                    day_status.append('nothing')

            if i == (len(self.df) - 1):
                if self.comission:
                    print(f'Комиссия за всё время(PSAR): {k}')
                    balance -= k

                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]

            count_df.append(count)
            balance_df.append(balance)
            price_df.append(price)

        psar_df["NumOfShares"] = count_df
        psar_df["MoneySpent"] = price_df
        psar_df["Balance"] = balance_df
        psar_df["DayStatus"] = day_status

        if plot:
            subplots_data = [
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'black'}
                    ],
                    'title': 'Цены закрытия акций',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#f0f0f0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'black'},
                        {'x': self.df['<DATE>'], 'y': self.df['PSARu'], 'label': 'PSAR Up', 'color': 'blue'},
                        {'x': self.df['<DATE>'], 'y': self.df['PSARd'], 'label': 'PSAR Down', 'color': 'orange'}
                    ],
                    'title': 'Индикатор PSAR',
                    'xlabel': 'Дата',
                    'ylabel': 'Stochastic',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'blue'}
                    ],
                    'scatter': [
                        {'x': self.df[self.df['Buy_Signal']].index, 'y': self.df[self.df['Buy_Signal']]['<CLOSE>'],
                         'label': 'Покупка', 'color': 'green', 'size': 100},
                        {'x': self.df[self.df['Sell_Signal']].index, 'y': self.df[self.df['Sell_Signal']]['<CLOSE>'],
                         'label': 'Продажа', 'color': 'red', 'size': 100}
                    ],
                    'title': 'Торговые сигналы на основе PSAR',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                }
            ]

            plot_data_with_subplots(subplots_data)

        return psar_df

    def rsi(self, plot=True):
        self.df["RSI"] = RSIIndicator(self.df['<CLOSE>'], 10).rsi()

        rsi_df = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            action = self.check_conditions_rsi(i, balance, count) if i != 0 else 'nothing'
            match action:
                case 'buy':
                    count += balance // self.df['<CLOSE>'][i]
                    price = self.df['<CLOSE>'][i] * count
                    balance = balance - price
                    k += price * 0.01
                    day_status.append('buy')
                    self.df.at[i, 'Buy_Signal'] = True

                case 'sell':
                    balance += count * self.df['<CLOSE>'][i]
                    k += count * self.df['<CLOSE>'][i] * 0.01
                    count = 0
                    price = self.df['<CLOSE>'][i]
                    day_status.append('sell')
                    self.df.at[i, 'Sell_Signal'] = True

                case 'nothing':
                    day_status.append('nothing')

            if i == (len(self.df) - 1):
                if self.comission:
                    print(f'Комиссия за всё время(RSI): {k}')
                    balance -= k

                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]

            count_df.append(count)
            balance_df.append(balance)
            price_df.append(price)

        rsi_df["NumOfShares"] = count_df
        rsi_df["MoneySpent"] = price_df
        rsi_df["Balance"] = balance_df
        rsi_df["DayStatus"] = day_status
        
        if plot:
            subplots_data = [
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'black'}
                    ],
                    'title': 'Цены закрытия акций',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#f0f0f0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['RSI'], 'label': 'RSI Indicator', 'color': 'red'}
                    ],
                    'title': 'Индикатор RSI',
                    'xlabel': 'Дата',
                    'ylabel': 'IVAR',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'blue'}
                    ],
                    'scatter': [
                        {'x': self.df[self.df['Buy_Signal']].index, 'y': self.df[self.df['Buy_Signal']]['<CLOSE>'],
                         'label': 'Покупка', 'color': 'green', 'size': 100},
                        {'x': self.df[self.df['Sell_Signal']].index, 'y': self.df[self.df['Sell_Signal']]['<CLOSE>'],
                         'label': 'Продажа', 'color': 'red', 'size': 100}
                    ],
                    'title': 'Торговые сигналы на основе RSI',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                }
            ]

            plot_data_with_subplots(subplots_data)
        
        return rsi_df

    def stochastic(self, window_fast=11, window_slow=27, window_sign=9, plot=True):
        stochastic = StochasticOscillator(high=self.df["<HIGH>"], low=self.df["<LOW>"], close=self.df["<CLOSE>"])
        self.df["Stoch"] = stochastic.stoch()
        self.df["StochSig"] = stochastic.stoch_signal()
        _MACD = MACD(self.df["<CLOSE>"], window_fast=window_fast, window_slow=window_slow, window_sign=window_sign)
        self.df["MACD"] = _MACD.macd()
        self.df["MACD_signal"] = _MACD.macd_signal()
        self.df["MACD_gist"] = _MACD.macd_diff()

        stochastic_df = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            action = self.check_conditions_stochastic(i, balance, count) if i != 0 else 'nothing'
            match action:
                case 'buy':
                    count += balance // self.df['<CLOSE>'][i]
                    price = self.df['<CLOSE>'][i] * count
                    balance = balance - price
                    k += price * 0.01
                    day_status.append('buy')
                    self.df.at[i, 'Buy_Signal'] = True

                case 'sell':
                    balance += count * self.df['<CLOSE>'][i]
                    k += count * self.df['<CLOSE>'][i] * 0.01
                    count = 0
                    price = self.df['<CLOSE>'][i]
                    day_status.append('sell')
                    self.df.at[i, 'Sell_Signal'] = True

                case 'nothing':
                    day_status.append('nothing')

            if i == (len(self.df) - 1):
                if self.comission:
                    print(f'Комиссия за всё время(Stochastic): {k}')
                    balance -= k

                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]

            count_df.append(count)
            balance_df.append(balance)
            price_df.append(price)

        stochastic_df["NumOfShares"] = count_df
        stochastic_df["MoneySpent"] = price_df
        stochastic_df["Balance"] = balance_df
        stochastic_df["DayStatus"] = day_status

        if plot:
            subplots_data = [
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'black'}
                    ],
                    'title': 'Цены закрытия акций',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#f0f0f0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['Stoch'], 'label': 'Stochastic Oscillator',
                         'color': 'blue'},
                        {'x': self.df['<DATE>'], 'y': self.df['StochSig'], 'label': 'Signal Stochastic Oscillator',
                         'color': 'orange'}
                    ],
                    'title': 'Индикатор Stochastic',
                    'xlabel': 'Дата',
                    'ylabel': 'Stochastic',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['MACD_signal'], 'label': 'Signal line', 'color': 'blue'},
                        {'x': self.df['<DATE>'], 'y': self.df['MACD'], 'label': 'MACD line', 'color': 'orange'}
                    ],
                    'bar': [
                        {'x': self.df['<DATE>'], 'y': self.df['MACD_gist'], 'label': 'MACD Histogram',
                         'color': self.df['MACD_gist'].apply(lambda x: '#ef5350' if x < 0 else '#26a69a')}
                    ],
                    'title': 'Индикатор MACD',
                    'xlabel': 'Дата',
                    'ylabel': 'MACD',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'blue'}
                    ],
                    'scatter': [
                        {'x': self.df[self.df['Buy_Signal']].index, 'y': self.df[self.df['Buy_Signal']]['<CLOSE>'],
                         'label': 'Покупка', 'color': 'green', 'size': 100},
                        {'x': self.df[self.df['Sell_Signal']].index, 'y': self.df[self.df['Sell_Signal']]['<CLOSE>'],
                         'label': 'Продажа', 'color': 'red', 'size': 100}
                    ],
                    'title': 'Торговые сигналы на основе Stochastic',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                }
            ]

            plot_data_with_subplots(subplots_data)

        return stochastic_df

    def ivar(self, window_size=10, plot=True):
        calc = [np.std(i) for i in self.df['<CLOSE>'].rolling(window=window_size)]
        ivar_values = (1 - calc / np.max(calc)).reshape(1, len(calc))[0]
        flat_flag = [0 if ivar_values[i] > 0.5 else 1 for i in range(len(ivar_values))]
        self.df['ivar'] = ivar_values
        self.df['flat_flag'] = flat_flag

        indicator_bb = BollingerBands(close=self.df["<CLOSE>"], window=20, window_dev=2)
        self.df['bb_bbm'] = indicator_bb.bollinger_mavg()
        self.df['bb_bbh'] = indicator_bb.bollinger_hband()
        self.df['bb_bbl'] = indicator_bb.bollinger_lband()
        self.df["SMA"] = SMAIndicator(close=self.df["<CLOSE>"], window=2).sma_indicator()

        ivar_df = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            action = self.check_conditions_ivar(i, balance, count) if i != 0 else 'nothing'
            match action:
                case 'buy':
                    count += balance // self.df['<CLOSE>'][i]
                    price = self.df['<CLOSE>'][i] * count
                    balance -= price
                    k += price * 0.01
                    day_status.append('buy')
                    self.df.at[i, 'Buy_Signal'] = True

                case 'sell':
                    balance += count * self.df['<CLOSE>'][i]
                    k += count * self.df['<CLOSE>'][i] * 0.01
                    count = 0
                    price = self.df['<CLOSE>'][i]
                    day_status.append('sell')
                    self.df.at[i, 'Sell_Signal'] = True

                case 'nothing':
                    day_status.append('nothing')

            if i == (len(self.df) - 1):
                if self.comission:
                    print(f'Комиссия за всё время(Ivar): {k}')
                    balance -= k

                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]

            count_df.append(count)
            balance_df.append(balance)
            price_df.append(price)

        ivar_df["NumOfShares"] = count_df
        ivar_df["MoneySpent"] = price_df
        ivar_df["Balance"] = balance_df
        ivar_df["DayStatus"] = day_status

        if plot:
            subplots_data = [
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'black'}
                    ],
                    'title': 'Цены закрытия акций',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#f0f0f0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['ivar'], 'label': 'IVAR', 'color': 'red'}
                    ],
                    'title': 'Индикатор IVAR',
                    'xlabel': 'Дата',
                    'ylabel': 'IVAR',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                },
                {
                    'lines': [
                        {'x': self.df['<DATE>'], 'y': self.df['<CLOSE>'], 'label': 'Цена закрытия', 'color': 'blue'}
                    ],
                    'scatter': [
                        {'x': self.df[self.df['Buy_Signal']].index, 'y': self.df[self.df['Buy_Signal']]['<CLOSE>'],
                         'label': 'Покупка', 'color': 'green', 'size': 100},
                        {'x': self.df[self.df['Sell_Signal']].index, 'y': self.df[self.df['Sell_Signal']]['<CLOSE>'],
                         'label': 'Продажа', 'color': 'red', 'size': 100}
                    ],
                    'title': 'Торговые сигналы на основе Ivar',
                    'xlabel': 'Дата',
                    'ylabel': 'Цена закрытия',
                    'grid': True,
                    'facecolor': '#e0e0e0'
                }
            ]

            plot_data_with_subplots(subplots_data)

        return ivar_df

    def _buy_and_hold(self):
        balance = self.balance
        count = balance // self.df['<CLOSE>'][0]
        price = self.df['<CLOSE>'][0] * count
        remaining = balance - price

        result = remaining + count * self.df['<CLOSE>'].values[-1]
        profit = result / balance - 1
        return np.round(profit, 4)

    def best_solution(self, plot_best=True):
        strategies = {
            'MACD+AROON': self.macd_aroon,
            'ATR+AROON': self.atr_aroon,
            'BB': self.bollinger_bands,
            'PSAR': self.psar,
            'RSI': self.rsi,
            'Stochastic': self.stochastic,
            'Ivar': self.ivar,
            # 'buy and hold': self._buy_and_hold
        }
        profitabilities = pd.DataFrame()
        profitabilities[''] = ['profitability, %']
        best_strat = None
        best_profit = -float('inf')
        best_name = None
        for strat, method in strategies.items():
            if strat == 'buy and hold':
                profitability = np.round(self._buy_and_hold(), 4)
            else:
                profitability = np.round(method(plot=False)['Balance'].values[-1] / self.balance - 1, 4)
            if profitability > best_profit:
                best_profit = profitability
                best_strat = method
                best_name = strat

            profitabilities[strat] = [profitability * 100]

        profitabilities = profitabilities.set_index([''])
        if plot_best:
            df = best_strat()
        else:
            df = best_strat(plot=False)
        colormap = lambda x: 'color: green' if x == best_profit * 100 else 'color: black'
        last_move = df['DayStatus'].values[-1]
        print(f"Лучшая стратегия: {best_name}")
        if last_move == 'buy':
            rec_move = 'Покупать'
        elif last_move == 'sell':
            rec_move = 'Продавать'
        else:
            rec_move = 'Удерживать'
        print(f"Рекоммендуемое действие: {rec_move}")
        np.savetxt(r'np.txt', df.values, fmt='%s')
        return profitabilities.style.map(colormap)
    
    def check_conditions_rsi(self, i, balance, count):
        close = self.df['<CLOSE>'][i]
        rsi_current = self.df["RSI"][i]
        rsi_previous = self.df["RSI"][i - 1]
        
        if balance > close and rsi_current < 30 < rsi_previous:
            return 'buy'
        elif count > 0 and rsi_current > 70:
            return 'sell'
        else:
            return 'nothing'        
    
    def check_conditions_atr_aroon(self, i, balance, count):
        close = self.df['<CLOSE>'][i]
        aroond_current = self.df['AROONd'][i]
        aroonu_current = self.df['AROONu'][i]
        atr_current = self.df["ATR"][i]
        atr_previous = self.df["ATR"][i - 1]
        smaatr_current = self.df["SMAATR"][i]
        smaatr_previous = self.df["SMAATR"][i - 1]

        if aroond_current <= 30 and aroonu_current >= 60:
            trend = 'up'
        elif aroonu_current <= 30 and aroond_current >= 60:
            trend = 'down'
        else:
            trend = None

        if balance > close and atr_current >= smaatr_current and trend == 'down':
            return 'buy'
        elif count > 0 and atr_current >= smaatr_current and atr_previous < smaatr_previous and trend == 'up':
            return 'sell'
        else:
            return 'nothing'

    def check_conditions_macd_aroon(self, i, balance, count):
        close = self.df['<CLOSE>'][i]
        aroond_current = self.df['AROONd'][i]
        aroonu_current = self.df['AROONu'][i]
        macd_current = self.df['MACD_gist'][i]

        if aroond_current <= 30 and aroonu_current >= 60:
            trend = 'up'
        elif aroonu_current <= 30 and aroond_current >= 60:
            trend = 'down'
        else:
            trend = None

        if balance >= close and macd_current > 0 and trend == 'up':
            return 'buy'
        elif count > 0 > macd_current and trend == 'down':
            return 'sell'
        else:
            return 'nothing'

    def check_conditions_psar(self, i, balance, count):
        close = self.df['<CLOSE>'][i]
        psard_current = self.df["PSARd"][i]
        psard_previous = self.df["PSARd"][i - 1]
        psaru_current = self.df["PSARu"][i]
        psaru_previous = self.df["PSARu"][i - 1]

        if balance >= close and pd.isna(psard_current) and not pd.isna(psard_previous):
            return 'buy'
        elif count > 0 and pd.isna(psaru_current) and not pd.isna(psaru_previous):
            return 'sell'
        else:
            return 'nothing'

    def check_conditions_bb(self, i, balance, count):
        close = self.df['<CLOSE>'][i]
        sma_current = self.df["SMA"][i]
        sma_previous = self.df["SMA"][i - 1]
        bb_current = self.df["bb_bbm"][i]
        bb_previous = self.df["bb_bbm"][i - 1]

        if balance >= close and sma_current >= bb_current and sma_previous < bb_previous:
            return 'buy'
        elif count > 0 and sma_current <= bb_current and sma_previous > bb_previous:
            return 'sell'
        else:
            return 'nothing'

    def check_conditions_stochastic(self, i, balance, count):
        close = self.df['<CLOSE>'][i]
        macd_current = self.df["MACD_gist"][i]
        stochastic_current = self.df["StochSig"][i]
        stochastic_previous = self.df["StochSig"][i - 1]

        if balance >= close and macd_current > 0 and stochastic_current > 80 > stochastic_previous:
            return 'buy'
        elif count > 0 > macd_current and stochastic_current < 20 < stochastic_previous:
            return 'sell'
        else:
            return 'nothing'

    def check_conditions_ivar(self, i, balance, count):
        close = self.df['<CLOSE>'][i]
        sma_current = self.df["SMA"][i]
        sma_previous = self.df["SMA"][i - 1]
        bbm_current = self.df["bb_bbm"][i]
        bbm_previous = self.df["bb_bbm"][i - 1]
        flat_flag = self.df['flat_flag'][i]

        if balance >= close and sma_current >= bbm_current and sma_previous < bbm_previous and flat_flag == 0:
            return 'buy'
        elif count > 0 and sma_current <= bbm_current and sma_previous > bbm_previous and flat_flag == 0:
            return 'sell'
        else:
            return 'nothing'


def plot_data_with_subplots(subplots_data, figure_size=(12, 8), main_xlabel='X-axis',
                            main_ylabel='Y-axis', legend=True, facecolor='#d8dcd6', save_plots=True):
    for subplot_index, subplot_data in enumerate(subplots_data):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figure_size, facecolor=facecolor)

        for line_data in subplot_data.get('lines', []):
            ax.plot(line_data['x'], line_data['y'], label=line_data.get('label', None),
                    color=line_data.get('color', 'black'))

        for scatter_data in subplot_data.get('scatter', []):
            ax.scatter(scatter_data['x'], scatter_data['y'], label=scatter_data.get('label', None),
                       color=scatter_data.get('color', 'black'), s=scatter_data.get('size', 50))

        for bar_data in subplot_data.get('bar', []):
            ax.bar(bar_data['x'], bar_data['y'], label=bar_data.get('label', None),
                   color=bar_data.get('color', 'black'), width=bar_data.get('width', 0.8))

        ax.set_title(subplot_data.get('title', ''))
        ax.set_xlabel(subplot_data.get('xlabel', main_xlabel))
        ax.set_ylabel(subplot_data.get('ylabel', main_ylabel))

        if legend and 'lines' in subplot_data and any('label' in line for line in subplot_data['lines']):
            ax.legend(loc='best')

        ax.grid(subplot_data.get('grid', True))
        ax.set_facecolor(subplot_data.get('facecolor', '#d8dcd6'))

        plt.tight_layout(pad=3.0)

        if save_plots:
            fig.savefig(f'subplot_{subplot_index}.png', dpi=100)

        plt.show(block=False)
    plt.show(block=True)


def main():
    load_dotenv()
    log = Logger(host=os.getenv('host'), db=os.getenv('database'), user=os.getenv('user'), password=os.getenv('password'),port=os.getenv('port') )
    res = Indicators(log,'GAZP', '06.08.2020', balance=100000, interval='day', days=365)
    res.best_solution()
    log.log_end()

if __name__ == '__main__':
    main()
