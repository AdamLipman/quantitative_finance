import numpy as np



class Option:

    def __init__(self, S0, K, T, rf, sigma, iterations):
        self.S0 = S0
        self.K = K
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_sim(self):
        # we need 2 columns: first with 0s, second will store payoff
        # we need the first column of 0s: payoff function is max(0,S-E) for a call
        call_option_data = np.zeros([self.iterations, 2])

        rand = np.random.normal(0, 1, [1, self.iterations])

        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.T) * rand)

        call_option_data[:, 1] = stock_price - self.K
        average = np.sum(np.amax(call_option_data, axis=1)) / float(self.iterations)

        # discount future price for present

        return np.exp(-1.0*self.rf*self.T)*average

    def put_option_sim(self):
        # we need 2 columns: first with 0s, second will store payoff
        # we need the first column of 0s: payoff function is max(0,S-E) for a call
        call_option_data = np.zeros([self.iterations, 2])

        rand = np.random.normal(0, 1, [1, self.iterations])

        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.T) * rand)

        call_option_data[:, 1] = self.K - stock_price
        average = np.sum(np.amax(call_option_data, axis=1)) / float(self.iterations)

        # discount future price for present

        return np.exp(-1.0*self.rf*self.T)*average


if __name__ == '__main__':
    option = Option(100, 100, 1, 0.005, 0.20, 1000)
    print('Call: ', option.call_option_sim())
    print('Put: ', option.put_option_sim())
