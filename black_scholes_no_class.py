from numpy import log, exp, sqrt, pi
from scipy import stats


def call(S, K, T, rf, sigma):
    d1 = (log(S / K) + (rf + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    return S * stats.norm.cdf(d1) - K * exp(-rf * T) * stats.norm.cdf(d2)


def put(S, K, T, rf, sigma):
    d1 = (log(S / K) + (rf + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    return -S * stats.norm.cdf(-d1) + K * exp(-rf * T) * stats.norm.cdf(-d2)

if __name__ == '__main__':
    print(call(100, 100, 1, 0.05, 0.2))
    print(put(100, 100, 1, 0.05, 0.2))

    #print(call(137.35, 137, 57/365, 0.005816, 0.3724))
    #print(put(137.35, 137, 57/365, 0.005816, 0.3724))
