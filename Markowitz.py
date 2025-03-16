import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
#import winsound


# sound for end of program
duration = 100  # ms
freq = 440  # Hz

num_trading_days = 252
# stocks we are going to handle

num_portfolios = 10000

stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

# historical data - define START and END dates

start_date = '2012-01-01'
end_date = '2017-01-01'


def download_data():
    # name of stock as key: stock values (2012-2017) as values
    stock_data = {}

    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(stock_data)


def show_data(data):
    data.plot(figsize=(10, 5))
    plt.show()


def calculate_return(data):
    log_return = np.log(data / data.shift(1))
    # print(log_return)
    return log_return[1:]


def show_statistics(returns):
    # instead of daily metrics we are after annual metrics
    print(returns.mean() * num_trading_days)
    print(returns.cov() * num_trading_days)


def show_mean_variance(returns):
    portfolio_return = np.sum(returns.mean() * weights) * num_trading_days
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * num_trading_days, weights)))
    print('Expected portfolio mean (return): ', portfolio_return)
    print('Expected portfolio volatility (std dev): ', portfolio_vol)


# using current 3 month treasury annualized
risk_free_rate = (1 + 0.8254 / 100) ** 4 - 1
print(risk_free_rate)


def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []
    sharpe_list = []

    for _ in range(num_portfolios):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * num_trading_days)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * num_trading_days, w))))
        sharpe_list.append((np.sum(returns.mean() * w) * num_trading_days - risk_free_rate) / np.sqrt(
            np.dot(w.T, np.dot(returns.cov() * num_trading_days, w))))
    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks), np.array(sharpe_list)


def show_portfolios(returns, vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(vols, returns, c=(returns - risk_free_rate) / vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * num_trading_days
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * num_trading_days, weights)))
    return np.array([portfolio_return, portfolio_vol, (portfolio_return - risk_free_rate) / portfolio_vol])


# scipy optimize module can find min of given function. Min of -f(x) = max of f(x)
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


# finding point with highest Sharpe ratio
def optimize_portfolio(weights, returns):
    # constraint that sum of weights == 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    # weights can be 1 at most - 1 when 100% of the money is invested into a single stock
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds,
                                 constraints=constraints)


def print_optimal_portfolio(optimum, returns):
    print('Optimal portfolio: ', optimum['x'].round(4))
    print('Expected return, volatility and Sharpe ratio: ', statistics(optimum['x'].round(4), returns).round(3))


def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=(portfolio_rets - risk_free_rate) / portfolio_vols, marker='o', s=3)
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'r*', markersize=10)
    plt.show()


if __name__ == '__main__':
    dataset = download_data()
    # show_data(dataset)
    log_daily_returns = calculate_return(dataset)
    # show_statistics(log_daily_returns)

    pweights, means, risks, sharpes = generate_portfolios(log_daily_returns)

    # using max(sharpe) only optimizes sharpe but not return and vol? Max is also less than optimized value...
    print(max(sharpes))
    max_sharpe = max(sharpes)
    print(means[np.where(max_sharpe)])
    print(risks[np.where(max_sharpe)])
    # show_portfolios(means, risks)
    optimum = optimize_portfolio(pweights, log_daily_returns)
    print_optimal_portfolio(optimum, log_daily_returns)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks)
#winsound.Beep(freq, duration)
