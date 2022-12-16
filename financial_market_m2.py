# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
--------------------------------------------------------------------------------------------------------
- Imports
--------------------------------------------------------------------------------------------------------
"""

import yfinance as yf
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from pypfopt import expected_returns
from pypfopt import  risk_models
import seaborn as sb
from pypfopt.efficient_frontier import EfficientFrontier
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'
import scipy.optimize as sco

"""
--------------------------------------------------------------------------------------------------------
- Functions
--------------------------------------------------------------------------------------------------------
"""
def get_historical_data(ticker, start_date):
    """
    Parameters
    ----------
    ticker : String
        The symbol of the stock that we would like to get.
    start_date : String
        Start date.
    Returns
    -------
    Array
        Return the historical stock data.
    """
    # Pull Historical Data
    data = yf.download(ticker, start=start_date)
    # Calculate Daily Returns
    data['Daily Return'] = data['Adj Close'].pct_change()   
    return data.dropna()

# Normalize stock data based on initial price
def normalize(df):
    """
    Parameters
    ----------
    df : Dataframe
        Dataframe that we would like to normalize.

    Returns
    -------
    x : Dataframe
        Normalized Dataframe.

    """
    x = df.copy()
    for i in x.columns[1:]:
      x[i] = x[i]/x[i][0]
    return x

# Function to plot interactive plot
def interactive_plot(df, title):
    """
    Parameters
    ----------
    df : Dataframe
        Dataframe that we would like to plot.
    title : String
        Title of the final graph.

    Returns
    -------
    None, procedure.
    """
    fig = px.line(title = title)
    for i in df.columns[1:]:
        print(i)
        if(i == "IWV"):
            fig.add_scatter(x = df.index, y = df[i], name = i)
            fig['data'][5]['line']['width']=10
        else:
            fig.add_scatter(x = df.index, y = df[i], name = i)
    fig.show()

def sharpe_ratio(weights):
    """
    Parameters
    ----------
    weights : Array
        array of weights.

    Returns
    -------
    Float
        Return the sharpe ratio.
    """
    return (sum(list(ER.values()) * weights) / np.sqrt(np.dot(weights.T,np.dot(daily_cov_matrix,weights))))

"""
--------------------------------------------------------------------------------------------------------
- Get Historical data
--------------------------------------------------------------------------------------------------------
"""
#TRAINING DATA -> FROM 2022-09-29
TSN = get_historical_data("TSN", "2017-01-01")
CCJ = get_historical_data("CCJ", "2017-01-01")
BIIB = get_historical_data("BIIB", "2017-01-01")
BAM = get_historical_data("BAM", "2017-01-01")
RGR = get_historical_data("RGR", "2017-01-01")
IWV = get_historical_data("IWV", "2017-01-01")

#EVALUATE DATA -> FROM 2022-09-29
TSN_evaluate = get_historical_data("TSN", "2022-09-29")
CCJ_evaluate = get_historical_data("CCJ", "2022-09-29")
BIIB_evaluate = get_historical_data("BIIB", "2022-09-29")
BAM_evaluate = get_historical_data("BAM", "2022-09-29")
RGR_evaluate = get_historical_data("RGR", "2022-09-29")
IWV_evaluate = get_historical_data("IWV", "2022-09-29")

"""
--------------------------------------------------------------------------------------------------------
- Compute STD & VAR
--------------------------------------------------------------------------------------------------------
"""
TSN_std = TSN["Daily Return"].std()
print("TSN std >>> ", TSN_std)
print("TSN var >>> ", TSN_std**2)

CCJ_std = CCJ["Daily Return"].std()
print("CCJ std >>> ", CCJ_std)
print("CCJ var >>> ", CCJ_std**2)

BIIB_std = BIIB["Daily Return"].std()
print("BIIB std >>> ", BIIB_std)
print("BIIB var >>> ", BIIB_std**2)

BAM_std = BAM["Daily Return"].std()
print("BAM std >>> ", BAM_std)
print("BAM var >>> ", BAM_std**2)

RGR_std = RGR["Daily Return"].std()
print("RGR std >>> ", RGR_std)
print("RGR var >>> ", RGR_std**2)

IWV_std = IWV["Daily Return"].std()
print("IWV std >>> ", IWV_std)
print("IWV var >>> ", IWV_std**2)

"""
--------------------------------------------------------------------------------------------------------
- Create stocks DataFrame
--------------------------------------------------------------------------------------------------------
"""
stocks = pd.DataFrame()
stocks["TSN"] = TSN.loc[:"2022-09-29 00:00:00"]["Close"]
stocks["CCJ"] = CCJ.loc[:"2022-09-29 00:00:00"]["Close"]
stocks["BIIB"] = BIIB.loc[:"2022-09-29 00:00:00"]["Close"]
stocks["BAM"] = BAM.loc[:"2022-09-29 00:00:00"]["Close"]
stocks["RGR"] = RGR.loc[:"2022-09-29 00:00:00"]["Close"]
stocks["IWV"] = IWV.loc[:"2022-09-29 00:00:00"]["Close"]

#Plot the stocks -> Normalized data in order to be able to compare them better
interactive_plot(normalize(stocks), 'Normalized Prices')

stocks_daily_return = pd.DataFrame()
stocks_daily_return["TSN"] = TSN.loc[:"2022-09-29 00:00:00"]["Daily Return"]
stocks_daily_return["CCJ"] = CCJ.loc[:"2022-09-29 00:00:00"]["Daily Return"]
stocks_daily_return["BIIB"] = BIIB.loc[:"2022-09-29 00:00:00"]["Daily Return"]
stocks_daily_return["BAM"] = BAM.loc[:"2022-09-29 00:00:00"]["Daily Return"]
stocks_daily_return["RGR"] = RGR.loc[:"2022-09-29 00:00:00"]["Daily Return"]
stocks_daily_return["IWV"] = IWV.loc[:"2022-09-29 00:00:00"]["Daily Return"]

"""
--------------------------------------------------------------------------------------------------------
- Expected returns of assets
--------------------------------------------------------------------------------------------------------
"""
#Risk free rate
rf = 0.0148 / 252
#Market free risk
mrp = 0.056 / 252

"""
--------------------------------------------------------------------------------------------------------
- Calculate beta of portfolio
--------------------------------------------------------------------------------------------------------
"""
beta = {}
alpha = {}

for i in stocks_daily_return.columns:
    stocks_daily_return.plot(kind = 'scatter', x = 'IWV', y = i)
    b, a = np.polyfit(stocks_daily_return['IWV'], stocks_daily_return[i], 1)
    plt.plot(stocks_daily_return['IWV'], b * stocks_daily_return['IWV'] + a, '-', color = 'r')  
    beta[i] = b
    alpha[i] = a  
    plt.show()

"""
--------------------------------------------------------------------------------------------------------
- Calculate CAPM for a Portfolio of Stocks
--------------------------------------------------------------------------------------------------------
"""
ER = {}
ER_list = []
keys = list(beta.keys())

for i in keys:
    if(i != 'IWV'):
        ER[i] = rf + (beta[i] * mrp)
        ER_list.append(rf + (beta[i] * mrp))
        
"""
--------------------------------------------------------------------------------------------------------
- Portfolio return of equaly weighted portfolio
--------------------------------------------------------------------------------------------------------
"""

portfolio = round(stocks_daily_return[["TSN", "CCJ", "BIIB", "BAM", "RGR"]], 6)
daily_cov_matrix = round(portfolio.cov(), 6)
yearly_cov_matrix = round(portfolio.cov(), 6) * 252

portfolio_weights = 1/5 * np.ones(5)
ER_portfolio = sum(list(ER.values()) * portfolio_weights)

daily_portfolio_var = np.dot(portfolio_weights.T,np.dot(daily_cov_matrix,portfolio_weights))
yearly_portfolio_var = np.dot(portfolio_weights.T,np.dot(yearly_cov_matrix,portfolio_weights))

sharpe_ratio(portfolio_weights)

"""
--------------------------------------------------------------------------------------------------------
- CAPM based optimization
--------------------------------------------------------------------------------------------------------
"""
capm_matrix = np.dot(np.linalg.inv(daily_cov_matrix), np.array(ER_list)-rf)
sum_capm_matrix = capm_matrix.sum()

"""
--------------------------------------------------------------------------------------------------------
- Compute equaly weighted portfolio
--------------------------------------------------------------------------------------------------------
"""

capm_weights = capm_matrix / sum_capm_matrix
ER_portfolio_capm = sum(list(ER.values()) * capm_weights)

"""
--------------------------------------------------------------------------------------------------------
- Efficient frontier method based on CAPM ERs
--------------------------------------------------------------------------------------------------------
"""     
portfolio_stds = portfolio.std()
portfolio_ers = pd.DataFrame(ER_list).set_index(portfolio_stds.index).astype(float)

assets = pd.concat([portfolio_ers, portfolio_stds], axis=1)
assets.columns = ['Returns', 'Volatility']

w = [0.2, 0.2, 0.2, 0.2, 0.2]
portfolio_er = (w*assets['Returns']).sum()
print("Portfolio expected return: ", portfolio_er*100, "%")

p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = len(portfolio.columns)
num_portfolios = 5000

for final_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, portfolio_ers)[0] # Returns are the product of individual expected returns of asset and its weigths
    p_ret.append(returns)
    var = np.dot(weights.T,np.dot(daily_cov_matrix,weights))# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(252) # Annual standard deviation = volatility
    p_vol.append(ann_sd)

data = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(portfolio.columns.tolist()):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in p_weights]
    
portfolios  = pd.DataFrame(data)
portfolios.head() # Dataframe of the 10000 portfolios created

"""
--------------------------------------------------------------------------------------------------------
- Plot efficient frontier
--------------------------------------------------------------------------------------------------------
"""

min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]

cum_return_port = pd.DataFrame()
cum_return_port["Port"] = stocks_daily_return["TSN"]*min_vol_port[2] + stocks_daily_return["CCJ"]*min_vol_port[3] + stocks_daily_return["BIIB"]*min_vol_port[4] + stocks_daily_return["BAM"]*min_vol_port[5] + stocks_daily_return["RGR"]*min_vol_port[6] 
cum_return_port["IWV"] = stocks_daily_return["IWV"]
cov_portfolio = cum_return_port.cov()
var_market = stocks_daily_return["IWV"].var()

0.000123 / var_market

"""
--------------------------------------------------------------------------------------------------------
- Find the optimal portfolio
--------------------------------------------------------------------------------------------------------
"""
rf = 0.0148 / 252 #risk factor
all_sharps = (portfolios['Returns']-rf)/portfolios['Volatility']
#	0.0009071174281580621
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]

portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10], facecolor='#494f51')
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='.', s=500)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='.', s=500)

evaluate_df = pd.DataFrame()
evaluate_df["TSN"] = TSN_evaluate["Close"]
evaluate_df["CCJ"] = CCJ_evaluate["Close"]
evaluate_df["BIIB"] = BIIB_evaluate["Close"]
evaluate_df["BAM"] = BAM_evaluate["Close"]
evaluate_df["RGR"] = RGR_evaluate["Close"]
evaluate_df["IWV"] = IWV_evaluate["Close"]

cum_return_port = pd.DataFrame()
cum_return_port["Port"] = stocks_daily_return["TSN"]*optimal_risky_port[2] + stocks_daily_return["CCJ"]*optimal_risky_port[3] + stocks_daily_return["BIIB"]*optimal_risky_port[4] + stocks_daily_return["BAM"]*optimal_risky_port[5] + stocks_daily_return["RGR"]*optimal_risky_port[6] 
cum_return_port["IWV"] = stocks_daily_return["IWV"]
cov_portfolio = cum_return_port.cov()
var_market = stocks_daily_return["IWV"].var()

0.000151 / var_market

interactive_plot(normalize(evaluate_df), 'Normalized Prices - Assets and Benchmark')

"""
--------------------------------------------------------------------------------------------------------
- Find the optimal portfolio
--------------------------------------------------------------------------------------------------------
"""
cum_return_port = pd.DataFrame()
cum_return_port["Port"] = TSN_evaluate["Daily Return"]*optimal_risky_port[2] + CCJ_evaluate["Daily Return"]*optimal_risky_port[3] + BIIB_evaluate["Daily Return"]*optimal_risky_port[4] + BAM_evaluate["Daily Return"]*optimal_risky_port[5] + RGR_evaluate["Daily Return"]*optimal_risky_port[6] 
cum_return_port["IWV"] = IWV_evaluate["Daily Return"]
cov_portfolio = cum_return_port.cov()
var_market = stocks_daily_return["IWV"].var()

cum_return_port["Port"].std()*np.sqrt(252)

0.000326 / var_market



ER = [0.000200, 0.000269, 0.000248, 0.000314, 0.000155]
weights = [0.185, 0.077, 0.120, 0.561, 0.058]
ER_portfolio_capm = sum(ER * weights)

ER = [0.00004, 0.00002, 0.00003, 0.00018, 0.00001]
sum(ER)*np.sqrt(252)












