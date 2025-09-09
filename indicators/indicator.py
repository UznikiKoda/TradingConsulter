
import matplotlib.pyplot as plt
import pandas as pd
import requests
import apimoex
import datetime
from dateutil.parser import parse
from abc import ABC
import numpy as np
from ta import add_all_ta_features

class Indicator(ABC):
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
    
    def get_tickers():
        url = 'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json?'
        data = requests.get(url).json()
        tickers = [row[0] for row in data['securities']['data']]
        return tickers

    def _buy_and_hold(self):
        balance = self.balance
        count = balance // self.df['<CLOSE>'][0]
        price = self.df['<CLOSE>'][0] * count
        remaining = balance - price

        result = remaining + count * self.df['<CLOSE>'].values[-1]
        profit = result / balance - 1
        return np.round(profit, 4)

    def plot_data_with_subplots(self, subplots_data, figure_size=(12, 8), main_xlabel='X-axis',
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
