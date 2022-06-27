import numpy as np
import yfinance as yf
import pandas as pd
import datetime


class VaR_monte:

    def __init__(self, stock, start_date, end_date):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date

    def download(self):
        data = {}
        ticker = yf.download(self.stock, self.start_date, self.end_date)
        data[self.stock] = ticker['Adj Close']
        return pd.DataFrame(data)

    def simulation(self, investment, mu, sigma, c, n, iterations):
        rand = np.random.normal(0, 1, [1, iterations])
        # random walk for stock price
        stock_price = investment * np.exp(
            n * (mu - 0.5 * sigma ** 2) + sigma * np.sqrt(n) * rand)
        # want to sort stock prices and find the price at the 1% lowest level to give 99% confidence stock wont go
        # that low
        stock_price = np.sort(stock_price)
        percentile = np.percentile(stock_price, (1 - c) * 100)
        return investment - percentile


if __name__ == '__main__':
    investment = 1e6
    c = 0.95
    n = 10
    iterations = 100000

    sample = VaR_monte('C', '2014-1-1', '2017-10-15')
    citi = sample.download()
    citi['returns'] = citi['C'].pct_change()
    citi = citi[1:]

    mu = np.mean(citi['returns'])
    sigma = np.std(citi['returns'])

    print('95% confidence that loss in investment will not exceed: $', round(sample.simulation(investment, mu, sigma, c, n, iterations), 3))
    print('This potential loss is {0:.3f}% of the total investment'.format(sample.simulation(investment, mu, sigma, c, n, iterations)/investment*100))
