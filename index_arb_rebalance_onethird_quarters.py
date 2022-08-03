import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import seaborn as sns
import scipy.optimize as optimization
import time

pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

stocks = ['QQQ', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOG', 'GOOGL', 'NVDA', 'META', 'PEP', 'COST']
num_trading_days_in_half_quarter = 21
num_portfolios = 100
num_half_quarters = 60

risk_free_rate = .025


class Arb:

    def __init__(self, ticks, start, end):
        self.ticks = ticks
        self.start = start
        self.end = end

    # def tic(self):
    #    return time.clock()

    # download all data
    def download(self):
        data = {}
        for stock in self.ticks:
            ticker = yf.download(stock, self.start, self.end)
            data[stock] = ticker['Adj Close']
        return pd.DataFrame(data)

    # heatmap looking at correlation coefficients
    def check_correlation(self):
        data = self.download()
        corrs = data.corr()
        sns.heatmap(corrs, xticklabels=corrs.columns, yticklabels=corrs.columns, vmin=-1, center=0, vmax=1, cmap='PuOr',
                    annot=True)
        plt.show()

    # the correlations are naturally all very close to 1. With aapl and msft = 23% of qqq and their correlation at 0.98, I will throw
    # out COST that has 0.98 with MSFT.

    # this method calculates a daily return dataframe for all assets
    def calculate_return(self):
        data = self.download()
        data = data.drop('COST', axis=1)
        data = data.drop('META', axis=1)
        data = data.drop('GOOG', axis=1)
        data = data.drop('GOOGL', axis=1)
        log_return = np.log(data / data.shift(1))
        # return_percent = data.pct_change()
        # print(log_return)
        # quarterly_returns = np.average(np.array(log_return[1:]).reshape(-1, num_trading_days_quarter), axis=0)
        quarterly_returns = log_return[1:].groupby(
            np.arange(len(log_return[1:])) // num_trading_days_in_half_quarter).mean()
        # quarterly_returns = pd.DataFrame(np.array(log_return[1:]).reshape(74, 119).mean(axis=0).reshape(17, 7))
        # quarterly_returns = pd.DataFrame(np.array(log_return[1:]).reshape(74, -1))
        # quarterly_returns.columns = ['QQQ', 'AAPL', 'TSLA', 'AMZN', 'MSFT', 'NVDA', 'PEP']
        return log_return[1:], quarterly_returns

    # I want the portfolio to be long stocks with weights and short qqq with weight = 1
    # For rebalancing, perform same operations as original script using one quarterly return. Returns should
    # be additive. For covariance matrix, use quarterly returns and optimal weights
    def generate_portfolios(self):
        dreturns, qreturns = self.calculate_return()
        # returns['QQQ'] = returns['QQQ'].apply(lambda x: x * -1)
        portfolio_means = []
        portfolio_risks = []
        portfolio_weights = []
        sharpe_list = []

        # len(stocks) - 4 since qqq is always -1 and I dropped 4 stocks
        for i in range(num_half_quarters):
            for _ in range(num_portfolios):
                w = np.random.random(len(stocks) - 5)
                w /= np.sum(w)
                # I'll insert a weight of -1 as the first position in w for qqq
                w = np.insert(w, 0, -1.0)
                portfolio_weights.append(w)
                portfolio_means.append(np.sum(qreturns.iloc[i] * w) * num_trading_days_in_half_quarter)
                portfolio_risks.append(
                    np.sqrt(np.dot(w.T, np.dot(dreturns.cov() * num_trading_days_in_half_quarter, w))))
                sharpe_list.append(
                    (np.sum(qreturns.iloc[i] * w) * num_trading_days_in_half_quarter - risk_free_rate / 12) / np.sqrt(
                        np.dot(w.T, np.dot(dreturns.cov() * num_trading_days_in_half_quarter, w))))
        # return 'weights', np.array(portfolio_weights).shape, 'returns', np.array(portfolio_means).shape, 'risks', np.array(portfolio_risks).shape, 'sharpes', np.array(sharpe_list).shape
        # portfolio_means = np.reshape(portfolio_means, (170, 1)).T
        # portfolio_risks = np.reshape(portfolio_risks, (170, 1)).T
        # sharpe_list = np.reshape(sharpe_list, (170, 1)).T
        return portfolio_weights, pd.DataFrame(
            {'returns': portfolio_means, 'risks': portfolio_risks, 'sharpes': sharpe_list})

    # scipy optimize module can find min of given function. Min of -f(x) = max of f(x)
    def max_function_sharpe(self):
        weights, data = self.generate_portfolios()
        max_sharpe_stats = data.sharpes.groupby(np.arange(len(data)) // num_portfolios).max()
        max_sharpe_indices = []
        optimal_weights = []
        sum_of_returns = 0
        sum_of_risks = 0
        sum_of_sharpes = 0
        for i in range(num_half_quarters):
            max_sharpe_indices.append(data.sharpes.tolist().index(max_sharpe_stats[i]))
            # for i in range(num_quarters):
            optimal_weights.append(weights[max_sharpe_indices[i]])
            sum_of_returns += data.returns[max_sharpe_indices[i]]
            sum_of_risks += data.risks[max_sharpe_indices[i]]
            sum_of_sharpes += data.sharpes[max_sharpe_indices[i]]
        return sum_of_returns, sum_of_risks, (
                sum_of_returns - risk_free_rate * 5) / sum_of_risks, optimal_weights

    def check_weight_correlation(self):
        return_tot, risk_tot, sharpe_tot, weights = self.max_function_sharpe()
        weights_df = pd.DataFrame(weights)
        weights_df.columns = ['QQQ', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'PEP']
        corrs = weights_df.corr()
        sns.heatmap(corrs, xticklabels=corrs.columns, yticklabels=corrs.columns, vmin=-1, center=0, vmax=1, cmap='PuOr',
                    annot=True)
        plt.show()
        # return weights_df

    def reg_weights(self):
        return_tot, risk_tot, sharpe_tot, weights = self.max_function_sharpe()
        weights_df = pd.DataFrame(weights)
        weights_df.columns = ['QQQ', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'PEP']
        reg_model = sm.OLS.from_formula('QQQ ~ AAPL+MSFT+AMZN+TSLA+NVDA+PEP', weights_df).fit()
        return reg_model.params

    # def show_optimal_portfolio(self):
    #    weights, portfolio_returns, portfolio_vols, sharpes, optimized_weights, optimized_returns, optimized_risk, optimized_sharpe = self.max_function_sharpe()
    #    plt.figure(figsize=(10, 6))
    #    plt.scatter(portfolio_vols, portfolio_returns, c=(portfolio_returns - risk_free_rate) / portfolio_vols,
    #                marker='o', s=3)
    #    plt.grid(True)
    #    plt.xlabel('Expected Volatility')
    #    plt.ylabel('Expected Return')
    #    plt.colorbar(label='Sharpe Ratio')
    #    plt.plot(optimized_risk, optimized_returns, 'r*', markersize=3)
    #    print(optimized_weights, optimized_returns, optimized_risk,
    #          optimized_sharpe)
    #    plt.show()

    # def toc(self):
    #    return time.clock()




if __name__ == '__main__':
    test = Arb(stocks, '2017-01-01', '2022-01-04')  # using 5.01 years so that length = 1260
    # print(test.tic())
    # print(test.download())
    # print(test.reg())
    # print(test.show_mean_variance())
    # test.check_correlation()
    # test.plot()
    # print(test.calculate_return())
    # print(test.show_statistics())
    # print(test.generate_portfolios())
    print(test.max_function_sharpe())
    # test.show_optimal_portfolio()
    # test.check_weight_correlation()
    # print(test.reg_weights())

    # print(test.toc())
