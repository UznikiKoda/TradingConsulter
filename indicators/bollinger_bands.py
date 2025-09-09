from .indicator import Indicator

import pandas as pd

from ta.volatility import BollingerBands
from ta.trend import SMAIndicator

class bollinger_bands(Indicator):
    def run(self, window=20, window_dev=2, plot=True):
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

            self.plot_data_with_subplots(subplots_data)

        return bb_df
    
    
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