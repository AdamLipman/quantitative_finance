import matplotlib.pyplot as plt
from numpy import log, exp, sqrt
from scipy import stats
import numpy as np


class Option:

    def __init__(self, S, K, T, rf, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.rf = rf
        self.sigma = sigma

    def calc_d1_d2(self):
        d1 = (log(self.S / self.K) + (self.rf + self.sigma ** 2 / 2) * self.T) / (self.sigma * sqrt(self.T))
        d2 = d1 - self.sigma * sqrt(self.T)

        return d1, d2

    def calc_N(self):
        x1, x2 = self.calc_d1_d2()

        N1p_ = stats.norm.cdf(x1)  # 1/sqrt(2pi) included in stats.norm definition
        N2p_ = stats.norm.cdf(x2)

        N1n_ = stats.norm.cdf(-x1)
        N2n_ = stats.norm.cdf(-x2)

        return N1p_, N2p_, N1n_, N2n_

    def calc_call_and_put(self):
        N1p, N2p, N1n, N2n = self.calc_N()
        call_price = self.S * N1p - self.K * exp(-self.rf * self.T) * N2p
        put_price = -self.S * N1n + self.K * exp(-self.rf * self.T) * N2n

        return call_price, put_price
        # print(put_price)

    def greeks(self):
        delta_C = self.calc_N()[0]
        delta_P = self.calc_N()[0] - 1
        gamma = stats.norm.pdf(self.calc_N()[0]) / (self.S * self.sigma * sqrt(self.T))
        vega = self.S * stats.norm.pdf(self.calc_N()[0]) * sqrt(self.T)/100 # /100 ?

        return delta_C, delta_P, gamma, vega


calls = []
puts = []
deltas_calls = []
deltas_puts = []
gammas = []
vegas = []
strikes = list(range(90, 111, 1))

for x in strikes:
    sample = Option(100, x, 3 / 365, 0.01, 0.30)
    calls.append(sample.calc_call_and_put()[0])
    puts.append(sample.calc_call_and_put()[1])
    deltas_calls.append(sample.greeks()[0])
    deltas_puts.append(sample.greeks()[1])
    gammas.append(sample.greeks()[2])
    vegas.append(sample.greeks()[3])


def plot_call_price_vs_strike():
    plt.plot(strikes, calls)
    plt.plot(strikes, puts)
    plt.xlabel('Strikes')
    plt.ylabel('Option Price')
    plt.show()


def plot_deltas():
    plt.plot(strikes, deltas_calls)
    plt.plot(strikes, deltas_puts)
    #plt.plot(strikes, np.array(deltas_calls) - np.array(deltas_puts))
    plt.plot(strikes, gammas)
    plt.xlabel('Strikes')
    plt.ylabel('Delta')
    plt.show()

def plot_gamma():
    plt.plot(strikes, gammas)
    plt.xlabel('Strikes')
    plt.ylabel('Gamma')
    plt.show()

def plot_vegas():
    plt.plot(strikes, vegas)
    plt.xlabel('Strikes')
    plt.ylabel('Vega')
    plt.show()


if __name__ == '__main__':
    # plot_call_price_vs_strike()
    plot_deltas()
    #plot_gamma()
    #plot_vegas()
