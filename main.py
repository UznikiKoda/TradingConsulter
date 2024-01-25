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


def get_tickers():
    url = 'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json?'
    data = requests.get(url).json()
    tickers = [row[0] for row in data['securities']['data']]
    return tickers


#todo: Настройки и построение графиков засунуть в отдельный def (в отдельный класс - файл)

class Indicators:
    def __init__(self, ticker, date, interval='day', days=365, balance=10 ** 5, comission=False):
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
        self.df.columns = ['<DATE>', '<OPEN>', '<CLOSE>', '<HIGH>', '<LOW>', '<VOL>']
        self.balance = balance
        self.df["<VOL>"] = pd.to_numeric(self.df["<VOL>"], errors='coerce')
        self.df["<VOL>"] = self.df["<VOL>"].fillna(0)
        self.df = add_all_ta_features(self.df, open="<OPEN>", high="<HIGH>", low="<LOW>", close="<CLOSE>",
                                      volume="<VOL>", fillna=True)

    def macd_aroon(self, window_fast=11, window_slow=27, window_sign=9, lb=25, plot=True):
        _MACD = MACD(self.df["<CLOSE>"], window_fast=11, window_slow=27, window_sign=9)
        self.df["MACD"] = _MACD.macd()
        self.df["MACD_signal"] = _MACD.macd_signal()
        self.df["MACD_gist"] = _MACD.macd_diff()
        self.df['AROONu'] = 100 * self.df['<HIGH>'].rolling(lb + 1).apply(lambda x: x.argmax()) / lb
        self.df['AROONd'] = 100 * self.df['<LOW>'].rolling(lb + 1).apply(lambda x: x.argmin()) / lb

        if plot:
            plt.figure(figsize=(40, 15), facecolor='#d8dcd6')
            ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
            ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1)

            ax1.plot(self.df['<DATE>'], self.df['MACD_signal'], label='Signal line')
            ax1.plot(self.df['<DATE>'], self.df['MACD'], label='MACD line')
            for i in range(len(self.df)):
                if str(self.df['MACD_gist'][i])[0] == '-':
                    ax1.bar(self.df['<DATE>'][i], self.df['MACD_gist'][i], color='#ef5350', )
                else:
                    ax1.bar(self.df['<DATE>'][i], self.df['MACD_gist'][i], color='#26a69a')

            ax2.plot(self.df['<DATE>'], self.df['AROONu'], label='AROON Up Channel')
            ax2.plot(self.df['<DATE>'], self.df['AROONd'], label='AROON Down Channel')

            ax1.set_title('MACD + Aroon')
            ax1.legend(loc='best')
            ax2.legend(loc='best')

            ax1.grid(True)
            ax2.grid(True)

            ax1.set_facecolor('#d8dcd6')
            ax2.set_facecolor('#d8dcd6')

            fig, ax = plt.subplots(facecolor="#d8dcd6", figsize=(40, 15))
            plt.plot(self.df['<DATE>'], self.df["<CLOSE>"])
            ax.set_facecolor('#d8dcd6')
            ax.grid(True)

        MACD_AROON = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            # Определение тренда
            if self.df['AROONd'][i] <= 30 and self.df['AROONu'][i] >= 60:
                trend = 'up'
            elif self.df['AROONu'][i] <= 30 and self.df['AROONd'][i] >= 60:
                trend = 'down'
            else:
                trend = None

            if balance >= self.df['<CLOSE>'][i] and self.df['MACD_gist'][i] > 0 and trend == 'up':
                num = balance // self.df['<CLOSE>'][i]
                count += balance // self.df['<CLOSE>'][i]
                price = self.df['<CLOSE>'][i] * num
                balance = balance - price
                k += price * 0.01
                day_status.append('buy')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='green', s=78)

            elif count > 0 and self.df['MACD_gist'][i] < 0 and trend == 'down':
                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]
                day_status.append('sell')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='red', s=78)

            else:
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

        MACD_AROON["NumOfShares"] = count_df
        MACD_AROON["MoneySpent"] = price_df
        MACD_AROON["Balance"] = balance_df
        MACD_AROON["DayStatus"] = day_status
        curr_move = MACD_AROON['DayStatus'].values[-1]
        return MACD_AROON

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

        if plot:
            plt.figure(figsize=(40, 15), facecolor='#d8dcd6')
            ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
            ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1)

            ax1.plot(self.df['<DATE>'], self.df['ATR'], label='ATR')
            ax1.plot(self.df['<DATE>'], self.df['SMAATR'], label='SMA for ATR')

            ax2.plot(self.df['<DATE>'], self.df['AROONu'], label='AROON Up Channel')
            ax2.plot(self.df['<DATE>'], self.df['AROONd'], label='AROON Down Channel')

            ax1.set_title('ATR + Aroon')
            ax1.legend(loc='best')
            ax2.legend(loc='best')

            ax1.grid(True)
            ax2.grid(True)

            ax1.set_facecolor('#d8dcd6')
            ax2.set_facecolor('#d8dcd6')

            fig, ax = plt.subplots(facecolor="#d8dcd6", figsize=(40, 15))
            plt.plot(self.df['<DATE>'], self.df["<CLOSE>"])

            ax.set_facecolor('#d8dcd6')
            ax.grid(True)

        ATR_AROON = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            # Определение тренда
            if self.df['AROONd'][i] <= 30 and self.df['AROONu'][i] >= 60:
                trend = 'up'
            elif self.df['AROONu'][i] <= 30 and self.df['AROONd'][i] >= 60:
                trend = 'down'
            else:
                trend = None

            if balance >= self.df['<CLOSE>'][i] and (self.df["ATR"][i] >= self.df["SMAATR"][i]) and trend == 'down':
                num = balance // self.df['<CLOSE>'][i]
                count += balance // self.df['<CLOSE>'][i]
                price = self.df['<CLOSE>'][i] * num
                balance = balance - price
                k += price * 0.01
                day_status.append('buy')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='green', s=78)

            elif count > 0 and (self.df["ATR"][i] >= self.df["SMAATR"][i]) and (
                    self.df["ATR"][i - 1] < self.df["SMAATR"][i - 1]) and trend == 'up':
                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]
                day_status.append('sell')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='red', s=78)

            else:
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

        ATR_AROON["NumOfShares"] = count_df
        ATR_AROON["MoneySpent"] = price_df
        ATR_AROON["Balance"] = balance_df
        ATR_AROON["DayStatus"] = day_status

        return ATR_AROON

    def bolinger_bands(self, window=20, window_dev=2, plot=True):
        indicator_bb = BollingerBands(close=self.df["<CLOSE>"], window=20, window_dev=2)
        self.df['bb_bbm'] = indicator_bb.bollinger_mavg()
        self.df['bb_bbh'] = indicator_bb.bollinger_hband()
        self.df['bb_bbl'] = indicator_bb.bollinger_lband()
        self.df["SMA"] = SMAIndicator(close=self.df["<CLOSE>"], window=2).sma_indicator()
        if plot:
            plt.figure(figsize=(40, 15), facecolor='#d8dcd6')
            ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)

            ax1.plot(self.df['<DATE>'], self.df['<CLOSE>'], label='Close', color='black')
            ax1.plot(self.df['<DATE>'], self.df['bb_bbm'], label='BB Middle')
            ax1.plot(self.df['<DATE>'], self.df['bb_bbh'], label='BB High')
            ax1.plot(self.df['<DATE>'], self.df['bb_bbl'], label='BB Low')
            ax1.plot(self.df['<DATE>'], self.df['SMA'], label='SMA', color='#a00498')

            ax1.set_title('BB')
            ax1.legend(loc='best')

            ax1.grid(True)

            ax1.set_facecolor('#d8dcd6')

            fig, ax = plt.subplots(facecolor="#d8dcd6", figsize=(40, 15))
            plt.plot(self.df['<DATE>'], self.df["<CLOSE>"])

            ax.set_facecolor('#d8dcd6')
            ax.grid(True)

        BB = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            price = 0
            if ((balance >= self.df['<CLOSE>'][i]) and (self.df["SMA"][i] >= self.df["bb_bbm"][i]) and (
                    self.df["SMA"][i - 1] < self.df["bb_bbm"][i - 1])):
                count += balance // self.df['<CLOSE>'][i]
                price = self.df['<CLOSE>'][i] * count
                balance = balance - price
                k += price * 0.01
                day_status.append('buy')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='green', s=78)

            elif count > 0 and (self.df["SMA"][i] <= self.df["bb_bbm"][i]) and (
                    self.df["SMA"][i - 1] > self.df["bb_bbm"][i - 1]):
                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]
                day_status.append('sell')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='red', s=78)

            else:
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

        BB["NumOfShares"] = count_df
        BB["MoneySpent"] = price_df
        BB["Balance"] = balance_df
        BB["DayStatus"] = day_status
        return BB

    def psar(self, step=0.02, max_step=0.2, plot=True):
        Psar = PSARIndicator(high=self.df["<HIGH>"], low=self.df["<LOW>"], close=self.df["<CLOSE>"], step=0.02,
                             max_step=0.2)
        self.df["PSARu"] = Psar.psar_up()
        self.df["PSARd"] = Psar.psar_down()
        if plot:
            plt.figure(figsize=(40, 15), facecolor='#d8dcd6')
            ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)

            ax1.plot(self.df['<DATE>'], self.df['<CLOSE>'], label='Close', color='black')
            ax1.scatter(self.df['<DATE>'], self.df['PSARu'], label='PSAR Up')
            ax1.scatter(self.df['<DATE>'], self.df['PSARd'], label='PSAR Down')

            ax1.set_title('PSAR Indicator')
            ax1.legend(loc='best')
            ax1.grid(True)
            ax1.set_facecolor('#d8dcd6')

            fig, ax = plt.subplots(facecolor="#d8dcd6", figsize=(40, 15))
            plt.plot(self.df['<DATE>'], self.df["<CLOSE>"])

            ax.set_facecolor('#d8dcd6')
            ax.grid(True)

        PSARdf = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            price = 0
            if i == 0:
                day_status.append('nothing')
            elif ((balance >= self.df['<CLOSE>'][i]) and (
                    pd.isna(self.df["PSARd"][i]) == True and pd.isna(self.df["PSARd"][i - 1]) == False)):
                count += balance // self.df['<CLOSE>'][i]
                price = self.df['<CLOSE>'][i] * count
                balance = balance - price
                k += price * 0.01
                day_status.append('buy')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='green', s=78)

            elif count > 0 and (pd.isna(self.df["PSARu"][i]) == True and pd.isna(self.df["PSARu"][i - 1]) == False):
                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]
                day_status.append('sell')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='red', s=78)

            else:
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

        PSARdf["NumOfShares"] = count_df
        PSARdf["MoneySpent"] = price_df
        PSARdf["Balance"] = balance_df
        PSARdf["DayStatus"] = day_status
        return PSARdf

    def rsi(self, plot=True):
        self.df["RSI"] = RSIIndicator(self.df['<CLOSE>'], 10).rsi()
        if plot:
            plt.figure(figsize=(40, 15), facecolor='#d8dcd6')
            ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
            ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1)

            ax1.plot(self.df['<DATE>'], self.df['<CLOSE>'], label='Close', color='black')
            ax2.plot(self.df['<DATE>'], self.df['RSI'], label='RSI Indicator')

            ax1.set_title('RSI Indicator')
            ax1.legend(loc='best')
            ax2.legend(loc='best')

            ax1.grid(True)
            ax2.grid(True)

            ax1.set_facecolor('#d8dcd6')
            ax2.set_facecolor('#d8dcd6')

            fig, ax = plt.subplots(facecolor="#d8dcd6", figsize=(40, 15))
            plt.plot(self.df['<DATE>'], self.df["<CLOSE>"])
            ax.set_facecolor('#d8dcd6')
            ax.grid(True)

        RSIdf = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            price = 0
            if i == 0:
                day_status.append('nothing')
            elif ((balance >= self.df['<CLOSE>'][i]) and (self.df["RSI"][i] < 30 and self.df["RSI"][i - 1] > 30)):
                count += balance // self.df['<CLOSE>'][i]
                price = self.df['<CLOSE>'][i] * count
                balance = balance - price
                k += price * 0.01
                day_status.append('buy')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='green', s=78)

            elif count > 0 and (self.df["RSI"][i] > 70):
                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]
                day_status.append('sell')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='red', s=78)

            else:
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

        RSIdf["NumOfShares"] = count_df
        RSIdf["MoneySpent"] = price_df
        RSIdf["Balance"] = balance_df
        RSIdf["DayStatus"] = day_status
        return RSIdf

    def stochastic(self, window_fast=11, window_slow=27, window_sign=9, plot=True):
        St = StochasticOscillator(high=self.df["<HIGH>"], low=self.df["<LOW>"], close=self.df["<CLOSE>"])
        self.df["Stoch"] = St.stoch()
        self.df["StochSig"] = St.stoch_signal()
        _MACD = MACD(self.df["<CLOSE>"], window_fast=11, window_slow=27, window_sign=9)
        self.df["MACD"] = _MACD.macd()
        self.df["MACD_signal"] = _MACD.macd_signal()
        self.df["MACD_gist"] = _MACD.macd_diff()

        if plot:
            plt.figure(figsize=(40, 15), facecolor='#d8dcd6')
            ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
            ax2 = plt.subplot2grid((11, 1), (5, 0), rowspan=3, colspan=1)
            ax3 = plt.subplot2grid((11, 1), (8, 0), rowspan=3, colspan=1)

            ax1.plot(self.df['<DATE>'], self.df['<CLOSE>'], label='Close', color='black')

            ax2.plot(self.df['<DATE>'], self.df['Stoch'], label='Stochastic Oscillator')
            ax2.plot(self.df['<DATE>'], self.df['StochSig'], label='Signal Stochastic Oscillator')

            ax3.plot(self.df['<DATE>'], self.df['MACD_signal'], label='Signal line')
            ax3.plot(self.df['<DATE>'], self.df['MACD'], label='MACD line')

            axs = [ax1, ax2, ax3]
            for i in range(len(self.df)):
                if str(self.df['MACD_gist'][i])[0] == '-':
                    ax3.bar(self.df['<DATE>'][i], self.df['MACD_gist'][i], color='#ef5350', )
                else:
                    ax3.bar(self.df['<DATE>'][i], self.df['MACD_gist'][i], color='#26a69a')

            ax1.set_title('Stochastic')
            for i in axs:
                i.legend(loc='best')
                i.grid(True)
                i.set_facecolor('#d8dcd6')

            fig, ax = plt.subplots(facecolor="#d8dcd6", figsize=(40, 15))
            plt.plot(self.df['<DATE>'], self.df["<CLOSE>"])
            ax.set_facecolor('#d8dcd6')
            ax.grid(True)

        Stochdf = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            price = 0
            if i == 0:
                day_status.append('nothing')
            elif ((balance >= self.df['<CLOSE>'][i]) and (
                    self.df["MACD_gist"][i] > 0 and self.df["StochSig"][i] > 80 and self.df["StochSig"][i - 1] < 80)):
                count += balance // self.df['<CLOSE>'][i]
                price = self.df['<CLOSE>'][i] * count
                balance = balance - price
                k += price * 0.01
                day_status.append('buy')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='green', s=78)

            elif count > 0 and (
                    self.df["MACD_gist"][i] < 0 and self.df["StochSig"][i] < 20 and self.df["StochSig"][i - 1] > 20):
                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]
                day_status.append('sell')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='red', s=78)

            else:
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

        Stochdf["NumOfShares"] = count_df
        Stochdf["MoneySpent"] = price_df
        Stochdf["Balance"] = balance_df
        Stochdf["DayStatus"] = day_status
        return Stochdf

    def ivar(self, window_size=10, plot=True):
        calc = [np.std(i) for i in self.df['<CLOSE>'].rolling(window=window_size)]
        ivar_values = (1 - calc / np.max(calc)).reshape(1, len(calc))[0]
        flat_flag = [0 if ivar_values[i] > 0.5 else 1 for i in range(len(ivar_values))]
        self.df['ivar'] = ivar_values
        self.df['flat_flag'] = flat_flag

        if plot:
            plt.figure(figsize=(40, 15), facecolor='#d8dcd6')
            ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
            ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1)

            ax1.plot(self.df['<DATE>'], self.df['<CLOSE>'], label='Close', color='black')
            ax2.plot(self.df['<DATE>'], self.df['ivar'], label='iVar')

            ax1.set_title('iVar')
            ax1.legend(loc='best')
            ax2.legend(loc='best')

            ax1.grid(True)
            ax2.grid(True)

            ax1.set_facecolor('#d8dcd6')
            ax2.set_facecolor('#d8dcd6')

            fig, ax = plt.subplots(facecolor="#d8dcd6", figsize=(40, 15))
            plt.plot(self.df['<DATE>'], self.df["<CLOSE>"])
            ax.set_facecolor('#d8dcd6')
            ax.grid(True)

        indicator_bb = BollingerBands(close=self.df["<CLOSE>"], window=20, window_dev=2)
        self.df['bb_bbm'] = indicator_bb.bollinger_mavg()
        self.df['bb_bbh'] = indicator_bb.bollinger_hband()
        self.df['bb_bbl'] = indicator_bb.bollinger_lband()
        self.df["SMA"] = SMAIndicator(close=self.df["<CLOSE>"], window=2).sma_indicator()

        Ivardf = pd.DataFrame(self.df[['<DATE>', '<CLOSE>']])
        balance = self.balance
        price = 0
        count = 0
        k = 0
        balance_df, count_df, price_df, day_status = [], [], [], []

        for i in range(len(self.df)):
            price = 0
            if ((balance >= self.df['<CLOSE>'][i]) and (self.df["SMA"][i] >= self.df["bb_bbm"][i]) and (
                    self.df["SMA"][i - 1] < self.df["bb_bbm"][i - 1])) and self.df['flat_flag'][i] == 0:
                count += balance // self.df['<CLOSE>'][i]
                price = self.df['<CLOSE>'][i] * count
                balance = balance - price
                k += price * 0.01
                day_status.append('buy')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='green', s=78)

            elif count > 0 and (self.df["SMA"][i] <= self.df["bb_bbm"][i]) and (
                    self.df["SMA"][i - 1] > self.df["bb_bbm"][i - 1]) and self.df['flat_flag'][i] == 0:
                balance += count * self.df['<CLOSE>'][i]
                k += count * self.df['<CLOSE>'][i] * 0.01
                count = 0
                price = self.df['<CLOSE>'][i]
                day_status.append('sell')
                if plot:
                    plt.scatter(self.df['<DATE>'][i], self.df['<CLOSE>'][i], c='red', s=78)

            else:
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

        Ivardf["NumOfShares"] = count_df
        Ivardf["MoneySpent"] = price_df
        Ivardf["Balance"] = balance_df
        Ivardf["DayStatus"] = day_status
        return Ivardf

    def _buy_and_hold(self):
        balance = self.balance
        count = balance // self.df['<CLOSE>'][0]
        price = self.df['<CLOSE>'][0] * count
        remaining = balance - price

        # last selling
        result = remaining + count * self.df['<CLOSE>'].values[-1]
        profit = result / balance - 1
        return np.round(profit, 4)

    def best_solution(self, plot_best=True):
        strategies = {
            'MACD+AROON': self.macd_aroon,
            'ATR+AROON': self.atr_aroon,
            'BB': self.bolinger_bands,
            'PSAR': self.psar,
            'RSI': self.rsi,
            'Stochastic': self.stochastic,
            'Ivar': self.ivar,
            'buy and hold': self._buy_and_hold
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
        plt.savefig('best_solution', dpi = 100)
        return profitabilities.style.map(colormap)


def main():
    res = Indicators('GAZP', '06.08.2020', balance=100000, interval='day', days=365)
    res.best_solution()


if __name__ == '__main__':
    main()