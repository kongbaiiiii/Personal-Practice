from scipy import stats
from numpy import log, sqrt, exp

def call_price(S,E,T,rf,sigma):
    d1 = (log(S/E) + (rf + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    print('The d1 and d2 values are:', d1, d2)
    call = S*stats.norm.cdf(d1) - E*exp(-rf*T)*stats.norm.cdf(d2)
    return call

def put_price(S,E,T,rf,sigma):
    d1 = (log(S/E) + (rf + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    print('The d1 and d2 values are:', d1, d2)
    put = E*exp(-rf*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
    return put

if __name__ == "__main__":
    S = 100  # Underlying asset price
    E = 100  # Strike price
    T = 1    # Time to maturity (1 year)
    rf = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility

    print("Call Price:", call_price(S, E, T, rf, sigma))
    print("Put Price:", put_price(S, E, T, rf, sigma))