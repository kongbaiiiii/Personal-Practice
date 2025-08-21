import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# Define the stock universe
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'WMT']
start_date = '2012-01-01'
end_date = '2017-01-01'

def download_data(stocks, start_date, end_date):
    stock_data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(stock_data)

def plot_data(data):
    data.plot(figsize=(10, 5))
    plt.title('Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.show()
    
def calculate_return(data):
    returns = np.log(data / data.shift(1))
    return returns[1:]

def show_statistics(returns):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    print("Mean Annual Returns:\n", mean_returns)
    print("\nCovariance Matrix:\n", cov_matrix)
    return mean_returns, cov_matrix

def statistics(returns, weights):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])

def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    print(f"Portfolio Return: {portfolio_return:.2f}")
    print(f"Portfolio Volatility: {portfolio_volatility:.2f}")
    
def generate_portfolio(returns,num_portfolios=10000):
    num_assets = len(returns.columns)
    portfolio_weights = []
    portfolio_mean = []
    portfolio_volatility = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_weights.append(weights)

        portfolio_mean.append(np.sum(returns.mean() * weights) * 252)
        portfolio_volatility.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))))

    return np.array(portfolio_weights), np.array(portfolio_mean), np.array(portfolio_volatility)

def plot_portfolio(returns,volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, cmap='viridis', marker='o')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Portfolio Return vs Volatility')
    plt.grid(True)
    plt.show()

def min_sharpe_ratio(weights, returns):
    return -statistics(returns, weights)[2]  # Negative Sharpe Ratio for minimization
    
def optimize_portfolio(returns, weights):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(returns.columns)))
    result = optimize.minimize(fun=min_sharpe_ratio, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def print_optimal_portfolio(optimum, returns):
    optimal_weights = optimum['x'].round(4)
    portfolio_return, portfolio_volatility, sharpe_ratio = statistics(returns, optimal_weights)
    print(f"Optimal Weights: {optimal_weights}")
    print(f"Expected Portfolio Return: {portfolio_return:.2f}")
    print(f"Expected Portfolio Volatility: {portfolio_volatility:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
def plot_optimal_portfolio(opt, returns, portfolio_return, portfolio_volatility):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_volatility, portfolio_return, c=portfolio_return / portfolio_volatility, cmap='viridis', marker='o')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(statistics(returns, opt['x'])[1], statistics(returns, opt['x'])[0], color='red', marker='*', s=200, label='Optimal Portfolio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Portfolio Return vs Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # np.random.seed(42)  # For reproducibility
    stock_data = download_data(stocks, start_date, end_date)
    # print(stock_data.head())
    # plot_data(stock_data)
    
    returns = calculate_return(stock_data)
    # show_statistics(returns)

    weights, mean_returns, volatilities = generate_portfolio(returns)
    # plot_portfolio(mean_returns, volatilities)

    opt = optimize_portfolio(returns, weights)
    plot_optimal_portfolio(opt, returns, mean_returns, volatilities)
    print_optimal_portfolio(opt, returns)
