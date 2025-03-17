import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


num_of_sims = 1000


class Monte:
    result = []

    def __init__(self, S0, mu, sigma, N):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.N = N

    def sim_stock(self):
        for _ in range(num_of_sims):
            prices = [self.S0]
            for _ in range(self.N):
                # simulate the change day by day (t=1)
                stock_price = prices[-1] * np.exp(
                    (self.mu - 0.5 * self.sigma ** 2) * 1 + self.sigma * np.random.normal() * np.sqrt(1))
                prices.append(stock_price)

            self.result.append(prices)

        simulation_data = pd.DataFrame(self.result)
        # the columns will contain the time series for a given sim after transpose
        simulation_data = simulation_data.T
        return simulation_data

    def plot_data(self):
        sim_data = self.sim_stock()
        sim_data['mean'] = sim_data.mean(axis=1)
        # plt.plot(sim_data['mean'])
        plt.plot(sim_data)
        plt.show()


if __name__ == '__main__':
    sample = Monte(400, 0.000378, 0.005, 252)
    sample.plot_data()
