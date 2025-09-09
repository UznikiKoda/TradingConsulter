from .indicator import Indicator
import numpy as np
import pandas as pd
from ta.trend import SMAIndicator

class atr_aroon(Indicator):
    def run (self, window=14, lb=25, plot=True):
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

            self.plot_data_with_subplots(subplots_data)

        return atr_aroon_df

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
