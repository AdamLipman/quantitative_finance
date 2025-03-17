import numpy as np
import matplotlib.pyplot as plt


def sim_geo_rand_walk(S0, T=2000, N=1000, mu=0.001, sigma=0.005):
    dt = T / N
    t = np.linspace(0,T,N)
    # stnd norm dist with mean 0, std dev 1
    W = np.random.standard_normal(N)
    # N(0,dt) = sqrt(dt)*N(0,1)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.50*sigma**2)*t+sigma*W
    S = S0*np.exp(X)

    return t, S

def plot_sim(t, S):
    plt.plot(t, S)
    plt.xlabel('Time (t)')
    plt.ylabel('Stock Price S(t)')
    plt.title('Geometric Brownian Motion')
    plt.show()


if __name__ == '__main__':
    time, data = sim_geo_rand_walk(10)
    plot_sim(time, data)
