from ta.momentum import StochasticOscillator
from .indicator import Indicator
from ta.trend import MACD
import pandas as pd

class stochastic(Indicator):
    def run(self, window_fast=11, window_slow=27, window_sign=9, plot=True):
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

            self.plot_data_with_subplots(subplots_data)

        return stochastic_df
    
    
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