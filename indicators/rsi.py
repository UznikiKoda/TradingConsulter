from .indicator import Indicator
from ta.momentum import RSIIndicator
import pandas as pd

class RSI(Indicator):
    def run(self, plot=True):
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

            self.plot_data_with_subplots(subplots_data)
        
        return rsi_df
    
    
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