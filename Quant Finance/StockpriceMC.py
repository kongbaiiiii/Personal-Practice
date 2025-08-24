import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def stock_price_mc(S0, mu, sigma, n_scenarios=1000):
    result = []
    
    for _ in range(1000):
        prices = [S0]
        for _ in range(n_scenarios):
            stock_price = prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal())
            prices.append(stock_price)
        result.append(prices)
        
    data = pd.DataFrame(result).T
    data['Mean'] = data.mean(axis=1)
    
    plt.plot(data, alpha=0.1)
    plt.plot(data['Mean'], color='black', label='Mean Stock Price')
    plt.title('Monte Carlo Simulation of Stock Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    S0 = 50  # Initial stock price
    mu = 0.0002  # Expected return
    sigma = 0.01  # Volatility

    stock_price_mc(S0, mu, sigma)