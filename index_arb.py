import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import seaborn as sns
import datetime
import winsound

duration = 500  # milliseconds
freq = 440  # Hz

stocks = ['QQQ', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOG', 'GOOGL', 'NVDA', 'META', 'PEP', 'COST']

num_trading_days = 252
num_portfolios = 100000

risk_free_rate = .025


class Arb:

    def __init__(self, ticks, start, end, train_percent):
        self.ticks = ticks
        self.start = start
        self.end = end
        self.train_percent = train_percent

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
        data = data.drop(['COST', 'GOOG', 'GOOGL'], axis=1)
        log_return = np.log(data / data.shift(1))
        return log_return[1:]

    def show_statistics(self):
        # instead of daily metrics we are after annual metrics
        returns = self.calculate_return()
        print(returns.mean() * num_trading_days)
        print(returns.cov() * num_trading_days)

    # I want the portfolio to be long stocks with weights and short qqq with weight = -1
    def generate_portfolios(self):
        returns = self.calculate_return()
        # separate returns data according to train_percent
        train = returns.iloc[:round(self.train_percent * len(returns.index))]
        test = returns.iloc[round(self.train_percent * len(returns.index)):len(returns.index)]
        # returns['QQQ'] = returns['QQQ'].apply(lambda x: x * -1)
        portfolio_means = []
        portfolio_risks = []
        portfolio_weights = []
        sharpe_list = []

        # len(stocks) - 4 since qqq is always -1 and I dropped 3 stocks
        for _ in range(num_portfolios):
            w = np.random.random(len(stocks) - 4)
            w /= np.sum(w)
            # I'll insert a weight of 1 as the first position in w for qqq
            w = np.insert(w, 0, -1.0)
            portfolio_weights.append(w)
            portfolio_means.append(np.sum(train.mean() * w) * num_trading_days * self.train_percent)
            portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(train.cov() * num_trading_days * self.train_percent, w))))
            sharpe_list.append((np.sum(
                train.mean() * w) * num_trading_days * self.train_percent - risk_free_rate * self.train_percent) / np.sqrt(
                np.dot(w.T, np.dot(train.cov() * num_trading_days * self.train_percent, w))))
        return train, test, np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks), np.array(
            sharpe_list)

    # def show_mean_variance(self):
    #    returns = self.calculate_return()
    #    portfolio_return = np.sum(returns.mean() * weights) * num_trading_days
    #    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * num_trading_days, weights)))
    #    print('Expected portfolio mean (return): ', portfolio_return)
    #    print('Expected portfolio volatility (std dev): ', portfolio_vol)

    # def statistics(self):
    #    weights, portfolio_returns, portfolio_vols, sharpes = self.generate_portfolios()
    #    return np.array([portfolio_returns, portfolio_vols, (portfolio_returns - risk_free_rate) / portfolio_vols])

    # scipy optimize module can find min of given function. Min of -f(x) = max of f(x)
    def max_sharpe_test(self):
        train, test, weights, train_returns, train_vols, train_sharpes = self.generate_portfolios()
        max_sharpe_index = train_sharpes.argmax()
        best_sharpe = train_sharpes[max_sharpe_index]
        optimal_weights = weights[max_sharpe_index]
        best_return = train_returns[max_sharpe_index]
        best_risk = train_vols[max_sharpe_index]
        test_return = np.sum(test.mean() * optimal_weights) * num_trading_days * (1 - self.train_percent)
        test_vol = np.sqrt(np.dot(optimal_weights.T,
                                  np.dot(test.cov() * num_trading_days * (1 - self.train_percent), optimal_weights)))
        test_sharpe = (test_return - risk_free_rate * (1 - self.train_percent)) / test_vol
        return test_sharpe / best_sharpe, test_sharpe, best_sharpe, optimal_weights, self.train_percent

    # finding point with highest Sharpe ratio
    # def optimize_portfolio(self):
    #    weights, portfolio_returns, portfolio_vols, sharpes = self.generate_portfolios()
    #    function = self.min_function_sharpe()
    # constraint that sum of weights == 1
    #    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    # weights can be 1 at most - 1 when 100% of the money is invested into a single stock
    #    bounds = tuple((0, 1) for _ in range(len(stocks)-1))
    #    return optimization.minimize(fun=(portfolio_returns - risk_free_rate) / portfolio_vols, x0=weights[0], args=portfolio_returns, method='SLSQP',
    #                                 bounds=bounds,
    #                                 constraints=constraints)

    # def optimal_portfolio(self):
    #    optimum = self.optimize_portfolio()
    #    return 'Optimal portfolio: ', optimum['x'].round(4)
    # print('Expected return, volatility and Sharpe ratio: ', statistics(optimum['x'].round(4), returns).round(3))

    # note that calling generate portfolio here and looking at max_sharpe_test() will have different results due to RNG
    def show_optimal_portfolio(self):
        weights, portfolio_returns, portfolio_vols, sharpes, optimized_weights, optimized_returns, optimized_risk, optimized_sharpe = self.max_sharpe_test()
        plt.figure(figsize=(10, 6))
        plt.scatter(portfolio_vols, portfolio_returns, c=(portfolio_returns - risk_free_rate) / portfolio_vols,
                    marker='o',
                    s=3)
        plt.grid(True)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.plot(optimized_risk, optimized_returns, 'r*', markersize=3)
        return optimized_weights, optimized_returns, optimized_risk, optimized_sharpe
        # print(optimized_weights, optimized_returns, optimized_risk,
        #      optimized_sharpe)
        # plt.show()

    # def test_weights(self):
    #    train_weights, train_returns, train_risks, train_sharpe = self.show_optimal_portfolio()


if __name__ == '__main__':
    open('ratios_.txt', 'w').close()
    f = open('ratios_.txt', 'a')
    for r in range(6):
        qqq = Arb(stocks, str(datetime.date.today() - datetime.timedelta(num_trading_days * 365 / num_trading_days * 1)), str(datetime.date.today()), 0.8 + 0.02 * r)
    # qqq = Arb(stocks, '2017-01-01', '2022-01-01', 0.98)
    # print(qqq.download())
    # print(qqq.reg())
    # print(qqq.show_mean_variance())
    # qqq.check_correlation()
    # qqq.plot()
    # print(qqq.calculate_return())
    # qqq.show_statistics()
    # print(qqq.generate_portfolios())
    #for r in range(5):
        f.write('\n{}\n'.format(qqq.max_sharpe_test()))
    f.close()
    with open('ratios_.txt', 'r') as f:
        print(f.read())
    f.close()
    winsound.Beep(freq, duration)
    # print(qqq.show_optimal_portfolio())
    # print(qqq.max_sharpe_test(round(0.95 * len(qqq.calculate_return().index))))

