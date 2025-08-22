import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


class CAPM:
    
    def __init__(self,stocks, start_date, end_date):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def get_data(self):
        data = {}
        
        for stock in self.stocks:
            ticker = yf.download(stock, start=self.start_date, end=self.end_date, auto_adjust=False)
            data[stock] = ticker['Adj Close'] # Adjusted Close prices are used for CAPM analysis
            
        combined = pd.concat({k: v for k, v in data.items()}, axis=1)
        combined.columns = combined.columns.get_level_values(0)

        return combined

    def initialize(self):
        stock_data = self.get_data()
        stock_data = stock_data.resample('ME').last() # Resampling to monthly data

        self.data = pd.DataFrame({'s_adjclose': stock_data[self.stocks[0]], 'm_adjclose': stock_data[self.stocks[1]]})

        self.data[['s_returns', 'm_returns']] = np.log(self.data[['s_adjclose', 'm_adjclose']]
                                                         / self.data[['s_adjclose', 'm_adjclose']].shift(1))
        
        self.data = self.data[1:]  # Remove the first row with NaN values
    
    def calculate_beta(self):
        cov_matrix = np.cov(self.data['s_returns'], self.data['m_returns'])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        return beta
        
    def regression(self):
        beta, alpha = np.polyfit(self.data['m_returns'], self.data['s_returns'], deg=1)
        expected_return = 0.05 + beta * (self.data['m_returns'].mean() * 12 - 0.05)
        return expected_return, alpha, beta

    def plot(self, alpha, beta):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data['m_returns'], self.data['s_returns'], label='Data Points', alpha=0.5)
        plt.plot(self.data['m_returns'], alpha + beta * self.data['m_returns'], color='red', label='CAPM Line')
        plt.title('CAPM Regression')
        plt.xlabel('Market Returns')
        plt.ylabel('Stock Returns')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    capm = CAPM(['AAPL', '^GSPC'], '2012-01-01', '2017-01-01')
    capm.initialize()
    beta = capm.calculate_beta()
    print(f"Beta of AAPL with respect to S&P 500: {beta:.4f}")
    expected_return, alpha, beta = capm.regression()
    print(f"Expected Return: {expected_return:.4f}, Alpha: {alpha:.4f}, Beta: {beta:.4f}")
    capm.plot(alpha, beta)
    