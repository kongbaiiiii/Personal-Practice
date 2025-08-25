import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import datetime as dt

def get_date(stock,start,end):
    data = {}
    ticker = yf.download(stock, start=start, end=end, auto_adjust=False)
    data[stock] = ticker['Adj Close']
    combined = pd.concat({k: v for k, v in data.items()}, axis=1)
    combined.columns = combined.columns.get_level_values(0)
    return combined

# Value at Risk Calculation tomorrow
def calculate_var(position, mean, std_dev, confidence_level=0.95):
    v = norm.ppf(1 - confidence_level)
    var = position * (mean - v * std_dev)
    return var

# Value at Risk Calculation n days
def calculate_var_n_days(position, mean, std_dev, days, confidence_level=0.95):
    v = norm.ppf(1 - confidence_level)
    var = position * (days * mean - np.sqrt(days) * v * std_dev)
    return var


class VaRMonteCarlo:
    def __init__(self, stock, mean, std_dev, days, iterations, confidence_level):
        self.stock = stock
        self.mean = mean
        self.std_dev = std_dev
        self.days = days
        self.iterations = iterations
        self.confidence_level = confidence_level

    def simulate(self):
        # Simulate price paths
        rand = np.random.normal(0,1,(1,self.iterations))
        stock_price = self.stock * np.exp(self.days * (self.mean - 0.5 * self.std_dev**2) 
                                          + self.std_dev * np.sqrt(self.days) * rand)
        stock_price = np.sort(stock_price)
        s_prime = np.percentile(stock_price, 100 * (1 - self.confidence_level))
        
        return self.stock - s_prime

if __name__ == "__main__":
    stock_data = get_date('AAPL', dt.datetime(2014,1,1), dt.datetime(2018,1,1))

    stock_data['returns'] = np.log(stock_data['AAPL'] / stock_data['AAPL'].shift(1))
    stock_data = stock_data[1:]

    mean = np.mean(stock_data['returns'])
    std_dev = np.std(stock_data['returns'])

    # var = calculate_var(1e6, mean, std_dev, confidence_level=0.95)
    # print(f"Value at Risk (VaR) for AAPL: {var.round(2)}")
    # var_10_days = calculate_var_n_days(1e6, mean, std_dev, days=10, confidence_level=0.95)
    # print(f"10-Day Value at Risk (VaR) for AAPL: {var_10_days.round(2)}")

    mc_var = VaRMonteCarlo(stock=1e6, mean=mean, std_dev=std_dev, days=1, iterations=10000, confidence_level=0.95)
    var_mc = mc_var.simulate()
    print(f"10-Day Value at Risk (VaR) for AAPL using Monte Carlo: {var_mc.round(2)}")