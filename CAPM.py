import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

risk_free_rate = 0.05
months_in_year = 12

class CAPM:

    def __init__(self, stocks, start_date, end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        data = {}
        for stock in self.stocks:
            ticker = yf.download(stock, self.start_date, self.end_date)
            data[stock] = ticker['Close']

        return pd.DataFrame(data)

    def initialize(self):
        stock_data = self.download_data()
        # use monthly returns
        stock_data = stock_data.resample('M').last()
        # print(stock_data)

        self.data = pd.DataFrame({'IBM_close': stock_data[self.stocks[0]], 'SP500_close': stock_data[self.stocks[1]]})

        # log returns
        self.data[['IBM_returns', 'SP500_returns']] = np.log(
            self.data[['IBM_close', 'SP500_close']] / self.data[['IBM_close', 'SP500_close']].shift(1))
        self.data = self.data[1:]

    def calc_beta(self):
        # covariance matrix
        cov_matrix = np.cov(self.data['IBM_returns'], self.data['SP500_returns'])
        # beta is cov(r1,r2)/var(r2)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        print('Calculated beta is: ', beta)

    def regression(self):
        # use lin reg to fit a line to the data
        # [stock_returns, market_returns] - slope is the beta
        betar, alpha = np.polyfit(self.data['SP500_returns'], self.data['IBM_returns'], deg=1)
        print('Beta from regression is: ', betar)
        expected_return = risk_free_rate + betar * (self.data['SP500_returns'].mean() * months_in_year-risk_free_rate)
        print('Expected return: ', expected_return)
        self.plot_regression(alpha, betar)

    def plot_regression(self, alpha, betar):
        fig, axis = plt.subplots(1, figsize=(10, 5))
        axis.scatter(self.data['SP500_returns'], self.data['IBM_returns'], label='Data points')
        axis.plot(self.data['SP500_returns'], betar * self.data['SP500_returns'] + alpha, color='red', label='CAPM Line')
        plt.title('Capital Asset Pricing Model, finding alphas and betas')
        plt.xlabel('Market return', fontsize=14)
        plt.ylabel('Stock return')
        plt.text(0.05, 0.20, r'$r_s = \beta * r_m + \alpha$', fontsize=9)
        plt.legend()
        plt.grid(True)
        plt.show()




if __name__ == '__main__':
    capm = CAPM(['IBM', '^GSPC'], '2001-01-01', '2022-01-01')
    capm.initialize()
    capm.calc_beta()
    capm.regression()