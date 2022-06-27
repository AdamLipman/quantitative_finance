import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

class Hist:

    def __init__(self, stocks, start_date, end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download(self):
        data = {}
        for stock in self.stocks:
            ticker = yf.download(stock, self.start_date, self.end_date)
            data[stock] = ticker['Close']
        return pd.DataFrame(data)

    def returns(self):
        IBM_data = self.download()
        log_returns = np.log(IBM_data[['IBM']] / IBM_data[['IBM']].shift(1))
        self.data = log_returns[1:]


    def show_plot(self):
        plt.hist(self.data, bins=700)
        stock_var = self.data.var()
        stock_mean = self.data.mean()
        sigma = np.sqrt(stock_var)
        x = np.linspace(stock_mean - 5 * sigma, stock_mean + 5 * sigma, 100)
        plt.plot(x, norm.pdf(x, stock_mean, sigma))
        plt.show()




if __name__ == '__main__':

    mydata = Hist(['IBM'], '2010-01-01', '2020-01-01')
    mydata.download()
    mydata.returns()
    mydata.show_plot()


