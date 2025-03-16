import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime



class VaR:

    def __init__(self, stock, start_date, end_date):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        data_for_frame = {}
        ticker = yf.download(self.stock, self.start_date, self.end_date)
        data_for_frame[self.stock] = ticker['Adj Close']
        return pd.DataFrame(data_for_frame)

    # calc value at risk for any days n in future
    def calc_var(self, position, c, mu, sigma, n):
        alpha_value = norm.ppf(1 - c)
        var = position * (mu * n - sigma * np.sqrt(n) * alpha_value)
        return var


if __name__ == '__main__':
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2018, 1, 1)
    stock_data = VaR('C', start, end)
    # log daily returns
    data = stock_data.download_data()
    data['returns'] = np.log(stock_data.download_data()['C'] / stock_data.download_data()['C'].shift(1))
    data = data[1:]

    # invest $1 mn
    S = 1e6
    # confidence level = 0.95
    c = 0.99
    n = 20

    # assume daily returns are normally distributed
    mu = np.mean(data['returns'])
    sigma = np.std(data['returns'])

    print('Value at risk is: %0.2f' % stock_data.calc_var(S, c, mu, sigma, n))
