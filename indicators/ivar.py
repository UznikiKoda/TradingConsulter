from indicators.indicator import Indicator
import numpy as np
import pandas as pd 
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator

class Ivar(Indicator):
    
    def run(self, window_size=10, plot=True):
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

            self.plot_data_with_subplots(subplots_data)

        return ivar_df
    
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