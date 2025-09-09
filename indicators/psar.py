
from .indicator import Indicator
from ta.trend import PSARIndicator
import pandas as pd

class psar(Indicator):
    def run(self, step=0.02, max_step=0.2, plot=True):
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

            self.plot_data_with_subplots(subplots_data)

        return psar_df

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
