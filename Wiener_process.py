import numpy as np
import matplotlib.pyplot as plt


def wiener_process(dt = 0.10, x0 = 0, n = 1000):

    # W(t=0) = 0
    #initialize W(t) with zeros
    W = np.zeros(n+1)

    # create N+1 timesteps: t=0,1,2...N
    t = np.linspace(x0, n, n+1)

    # use cumulative sum: on every step the additional value is drawn from norm dist, mean = 0, var = dt ... N(0,dt)
    # also: N(0,dt) = sqrt[dt]*N(0,1)
    W[1: n+1] = np.cumsum(np.random.normal(0, np.sqrt(dt), n))

    return t, W


def plot_process(t, W):
    plt.plot(t, W)
    plt.xlabel('Time (t)')
    plt.ylabel('Wiener process W(t)')
    plt.title('Wiener process')
    plt.show()


if __name__ == '__main__':

    time, Wdata = wiener_process()
    plot_process(time, Wdata)
