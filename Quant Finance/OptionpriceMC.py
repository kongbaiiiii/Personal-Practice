import numpy as np

class OptionPriceMC:
    
    def __init__(self, S0, E, T, rf, sigma, n_simulations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.n_simulations = n_simulations

    def call_option(self):
        # Monte Carlo simulation for European call option pricing
        option_data = np.zeros([self.n_simulations, 2])
        rand = np.random.normal(0, 1, [1,self.n_simulations])   
        stock_price = self.S0 * np.exp((self.rf - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * rand)
        option_data[:,1] = stock_price - self.E
        avg = np.sum(np.amax(option_data, axis=1)) / float(self.n_simulations)
        # Discount back to present value
        return np.exp(-self.rf * self.T) * avg
    
    def put_option(self):
        # Monte Carlo simulation for European put option pricing
        option_data = np.zeros([self.n_simulations, 2])
        rand = np.random.normal(0, 1, [1,self.n_simulations])   
        stock_price = self.S0 * np.exp((self.rf - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * rand)
        option_data[:,1] = self.E - stock_price
        avg = np.sum(np.amax(option_data, axis=1)) / float(self.n_simulations)
        # Discount back to present value
        return np.exp(-self.rf * self.T) * avg

if __name__ == "__main__":
    model = OptionPriceMC(S0=100, E=100, T=1, rf=0.05, sigma=0.2, n_simulations=10000)
    print("Call Option Price:", model.call_option())
    print("Put Option Price:", model.put_option())