from ta.trend import MACD
from .indicator import Indicator
import pandas as pd

class MacdArron(Indicator):
    def run(self, window_fast=11, window_slow=27, window_sign=9, lb=25, plot=True):
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

        # Вынести в отдельный класс
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

            self.plot_data_with_subplots(subplots_data)

        return macd_aroon_df
    
    
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