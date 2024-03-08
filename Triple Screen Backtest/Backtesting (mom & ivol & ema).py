# -*- coding: utf-8 -*-

"""
Instructions: 

    1. Files required:
        a. Commodity Data.xlsx
        b. SPGSCITR Data.xlsx - S&P GSCI (Goldman-Sachs Commodity Index) Total Return daily indices
        c. F-F_RF_daily.xlsx - Fama-French daily risk-free rates
    2. Install module "TA-Lib"
    3. Line 27: change directory path and ensure all above files are in the directory path
"""

import os
import sys
import warnings
import time
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

start_time = time.time()
os.chdir('/Users/redmond1500/Downloads/')

def getData(fname, sheet_spot, index_name):
    df_data = pd.read_excel(fname, sheet_name = sheet_spot)
    df_data.set_index(index_name, inplace=True)
    return df_data

data = getData('Commodity Data.xlsx', 'Return indices', 'date')
names = data.columns.tolist() # store commodities' names



"""
SUMMARY STATISTICS
==============================================================================
"""
# obtain data description: count, mean, SD, min, max, quartiles
commo_describe = data.describe()
print(commo_describe.round(2))

# obtain first observation dates
first_obs = []
for col_index in range(0,len(data.columns)):
    first_obs.append(data.iloc[:,col_index].first_valid_index())

# create summary table
df_summary = pd.DataFrame(index = data.columns)
df_summary["1st obs date"] = first_obs # add first observation dates into summary table

# add other data descriptions into summary table
for row_index in range(0,len(commo_describe.index)):
    df_summary[str(commo_describe.index[row_index])] = commo_describe.iloc[row_index,:]

print(df_summary.round(2)) # display summary table with 2 d.p.

df_summary.to_csv('summary statistics.csv')



"""
COMMODITY MARKET FACTOR
==============================================================================
"""
# compute daily returns
rets = data/data.shift(1) - 1 # return = price(t)/price(t-1) - 1
rets = rets['1970-01-05':] # align start date

# compute "commodity market factor": returns of equally-weighted portfolio
EW_rets = rets.mean(axis = 1)
EW_cumrets = (1 + EW_rets).cumprod() # compute cummulative returns

# compute all commodities' monthly returns
data_startmonth = data.resample('MS').first().iloc[1:] # first prices of each month
data_endmonth = data.resample('MS').last().iloc[1:] # last prices of each month
mthlyrets = data_endmonth/data_startmonth - 1 

mth_index = data.resample('M').last().index

# relabel index of mthlyrets with end of month dates
mthlyrets = mthlyrets.reset_index(drop = True)
mthlyrets['date'] = mth_index[1:].values
mthlyrets = mthlyrets.set_index('date')

EW_mthlyrets = mthlyrets[12:].mean(axis = 1) # equally weighted average monthly returns
EW_cummthlyrets = (1 + EW_mthlyrets).cumprod() # compute monthly cumulative returns

# plot monthly cumulative returns and save plot
plot1 = EW_cummthlyrets.plot(figsize = (15, 7), yticks = range(0, 8), \
                             title = 'Monthly Cumulative Returns of Commodity Market Factor')
plot1.get_figure().savefig("cumrets (CMF).png")


"""
SETTING UP
==============================================================================
"""
SPGSCI = getData('SPGSCITR Data.xlsx', 'Sheet1', 'Date') # load SPGSCI indices
SPGSCI = SPGSCI.fillna(method = 'ffill') # fill NA values with previous indices
SPGSCI_rets = SPGSCI/SPGSCI.shift(1) - 1 # compute SPGSCI daily returns
SPGSCI_rets = SPGSCI_rets['1970-01-05':] # align start date

RF = getData('F-F_RF_daily.xlsx', 'F-F_RF_daily', 'date') # load risk-free rates
RF = RF.fillna(method = 'ffill')
RF = RF['1970-01-05':]/100 # convert rates from percentage to decimal forms



"""
COMPUTE TRIPLE-SCREEN STRATEGY SIGNALS

    - Formation period = 12 months
    - Holding period = 1 month
    - Screens:
        1. Time Series Momentum (Mom) - Long (short) commodities with positive
        (negative) cumulative excess return
        2. Idiosyncratic Volatility (IVol) - IVol is estimated by the residual
        standard error/RMSE of a Linear Regression model of commodities' daily
        returns on SPGSCI daily returns. Long (short) commodities with IVol smaller
        (larger) than the SD of SPGSCI daily returns.
        3. Exponential Moving Average Crossover (EMA) - EMA is taken with the formula
        [(Price_ytd - EMA_ytd) * (2/(1+N)) + EMA_ytd], which we have chosen 1 and 3 for N
        It places greater weight to more recent data, as compared to the 1/N weights on 
        simple moving average strategy
    - Weights on each screen:
        1. Mom = 2
        2. IVol = 1
        3. EMA = 2
==============================================================================
"""

" ------- MOMENTUM AND IDIOSYNCRATIC VOLATILITY ------- "
# empty dataframes to store values
mom = pd.DataFrame(index = mth_index[13:], columns = names) # for cumulative excess returns
ivol = pd.DataFrame(index = mth_index[13:], columns = names) # for idiosyncratic volatilities
ivol_SPGSCIsd = pd.DataFrame(index = mth_index[13:], columns = ['SPGSCI stdev']) # for SPGSCI SDs
mthlyRF = pd.DataFrame(index = mth_index[12:], columns = ['RF']) # for monthly risk-free rates

tot_loops = mthlyrets.shape[0] - 12 # number of holding periods

# first formation period
d1 = datetime.date(1970, 1, 1) # start date
d2 = d1.replace(year = d1.year + 1) # end date + 1 day

start_sigtime = time.time()
warnings.filterwarnings("ignore")

for i in range(tot_loops):
    
    # select data in formation period (from d1 to (d2 - 1 day))
    form_rets = rets[d1:(d2 - timedelta (days = 1))] # daily returns
    form_RF = RF[d1:(d2 - timedelta (days = 1))] # daily rfr
    form_cumRF = (1 + form_RF).cumprod().iloc[-1][0] - 1 # compute cumulative rfr in formation period
    
    # compute monthly risk-free rates
    # d3 = first date in the final month of the formation period
    if d2.month == 1:
        d3 = d2.replace(year = d2.year - 1).replace(month = 12)
    else:
        d3 = d2.replace(month = d2.month - 1)
    
    mthlyRF.iloc[i] = (1 + form_RF[d3:]).cumprod().iloc[-1][0] - 1
    
    "Momentum"
    mom.iloc[i] = (1 + form_rets).cumprod(axis = 0).iloc[-1] - 1 - form_cumRF # cumulative excess returns
    
    "Idiosyncratic Volatility"
    x = SPGSCI_rets[d1:(d2 - timedelta (days = 1))] # select SPGSCI returns in formation period
    ivol_SPGSCIsd.iloc[i] = x.std()[0] # SD of SPGSCI
    rmse = pd.DataFrame(index = [0], columns = names) # to store RMSE values of LR models
    
    # run Linear Regression and compute RMSE values (IVol) for each commodity
    for j in range(len(names)):
        
        y = form_rets.iloc[:, j] # select commodity j
        
        if y.isna().sum().sum() > 0:
            ivol.iloc[i, j] = np.nan # skip if any NA values present
        else:
            lm = LinearRegression()
            lm = lm.fit(x, y) # y = jth commodity's daily returns, x = SPGSCI daily returns
            y_pred = lm.predict(x) # y values on line of best fit
            ivol.iloc[i, j] = np.sqrt(mean_squared_error(y, y_pred)) # IVol = RMSE

    # redefine d1 and d2 dates to one month later
    if d1.month == 12:
        d1 = d1.replace(year = d1.year + 1).replace(month = 1)
        d2 = d2.replace(year = d2.year + 1).replace(month = 1)
    else:
        d1 = d1.replace(month = d1.month + 1)
        d2 = d2.replace(month = d2.month + 1)

# generate last monthly risk-free rate
d3 = datetime.date(2012, 12, 1)
mthlyRF.iloc[-1] = (1 + RF[d3:]).cumprod().iloc[-1][0] - 1

# Signals Generation
# Momentum: 1 (buy) for positive excess returns and -1 (sell) for negative excess returns
momBS = mom
momBS[momBS > 0] = 1
momBS[momBS < 0] = -1
# IVol: 1 (buy) for IVol > SD of SPGSCI, NA otherwise
ivolB = ivol.sub(ivol_SPGSCIsd['SPGSCI stdev'], axis = 0) # subtract respective SD of SPGSCI from IVol
ivolB[ivolB > 0] = np.nan
ivolB[ivolB < 0] = 1


" ------- EXPONENTIAL MOVING AVERAGE ------- "

# Calculate the EMA for both Short-Term EMA and Long-Term EMA (N=1,3)
df_price_mth = data.groupby([data.index.year,data.index.month]).last()
all_EMA_list_mth = list()
N = [1,3]
for period_index in N:
    multiplier = 2/(period_index+1)
    df_EMA_mth = pd.DataFrame(index=df_price_mth.index[period_index-1:len(df_price_mth.index)+1], \
                              columns = df_price_mth.columns)
    EMA_count = 1
    for col_index in range(0,len(df_price_mth.columns)):
        EMA_list_mth = list()
        for row_index in range(0,len(df_price_mth.index)-period_index+1):
            if EMA_count == 1:
                EMA = df_price_mth.iloc[row_index:row_index+period_index,col_index].mean()
            else:
                EMA = (df_price_mth.iloc[row_index,col_index] - EMA) * multiplier + EMA
            EMA_list_mth.append(EMA)
        df_EMA_mth.iloc[:,col_index] = EMA_list_mth
    all_EMA_list_mth.append(df_EMA_mth)

# Use Short-Term EMA minus the Long-Term EMA to generate signals
ST_EMA_mth = all_EMA_list_mth[0]
LT_EMA_mth = all_EMA_list_mth[1]
ST_period_index = N[0]
LT_period_index = N[1]
df_diff_mth = ST_EMA_mth - LT_EMA_mth
position_EMA_mth = pd.DataFrame(index = df_diff_mth.index[LT_period_index-ST_period_index+1:], \
                                columns = df_diff_mth.columns)
    # If the difference is positive, a BUY signal would be recorded, vice versa
for col_index in range(0, len(position_EMA_mth.columns)):
    if df_diff_mth.iloc[LT_period_index-ST_period_index:,col_index].isnull().values.any() == True:
        start_row = df_diff_mth.iloc[:,col_index].index.\
        get_loc(df_diff_mth.iloc[:,col_index][df_diff_mth.iloc[:,col_index].notnull()].index[0])
        position_EMA_mth.iloc[start_row-(LT_period_index-ST_period_index):,col_index] = \
        df_diff_mth.iloc[start_row:len(df_diff_mth)-1,col_index].apply(lambda x: 1 if x >= 0 else -1).values
    else:
        position_EMA_mth.iloc[:,col_index] = df_diff_mth.iloc\
        [LT_period_index-ST_period_index:len(df_diff_mth)-1,col_index].apply(lambda x: 1 if x >= 0 else -1).values

# EMABS is the EMA position dataframefor combination
EMABS = position_EMA_mth.iloc[10:].reset_index(drop = True)
EMABS['date'] = mth_index[13:].values
EMABS = EMABS.set_index('date')


"""
RETURNS
==============================================================================
"""
### returns of individual screens ###
momrets = momBS * mthlyrets.iloc[12:]
ivolrets = ivolB * mthlyrets.iloc[12:]
EMArets = EMABS * mthlyrets.iloc[12:]
# cumulative returns of individual screens
momcumrets = pd.DataFrame(index = momrets.index)
momcumrets['EW Returns'] = momrets.mean(axis = 1) # equally weighted average returns
momcumrets['Cumulative Returns'] = (1 + momcumrets['EW Returns']).cumprod() 
ivolcumrets = pd.DataFrame(index = ivolrets.index) 
ivolcumrets['EW Returns'] = ivolrets.mean(axis = 1) 
ivolcumrets['Cumulative Returns'] = (1 + ivolcumrets['EW Returns']).cumprod()
EMAcumrets = pd.DataFrame(index = EMArets.index) 
EMAcumrets['EW Returns'] = EMArets.mean(axis = 1) 
EMAcumrets['Cumulative Returns'] = (1 + EMAcumrets['EW Returns']).cumprod()

### returns of portfolio (combining 3 screens' signals) ###
momBS_noNA = momBS.fillna(0)
ivolB_noNA = ivolB.fillna(0)
EMABS_noNA = EMABS.fillna(0)
totBS = 2*momBS_noNA + ivolB_noNA + 2*EMABS_noNA # weighted sum of signals
totBS['Sum of Signals'] = totBS.abs().sum(axis = 1) # sum of absolute signals of each row
portBS = totBS.iloc[:, :24].div(totBS['Sum of Signals'], axis = 0) # weighted signals to be used as weights in portfolio

portrets = portBS * mthlyrets.iloc[12:] 
portcumrets = pd.DataFrame(index = portrets.index) 
portcumrets['Returns'] = portrets.sum(axis = 1) 
portcumrets['Cumulative Returns'] = (1 + portcumrets['Returns']).cumprod()



"""
PLOTS
==============================================================================
"""
# combine all cumulative returns
allcumrets = pd.concat([EW_cummthlyrets, momcumrets['Cumulative Returns'], \
                        ivolcumrets['Cumulative Returns'], EMAcumrets['Cumulative Returns'], \
                        portcumrets['Cumulative Returns']], axis = 1)
allcumrets.columns = ['Commodity Market Factor','Momentum', 'IVol', 'EMA', 'Combined']
# add a row of 1's at the beginning (assume $1 initial investment)
base = pd.DataFrame({'Commodity Market Factor':[1],'Momentum':[1], 'IVol':[1], \
                     'EMA':[1], 'Combined':[1]}, index = mth_index[12:13])
allcumrets = pd.concat([base, allcumrets], axis = 0) 

# plot cumulative returns and save plots
plot2 = allcumrets.iloc[:,0:4].plot(figsize = (15, 7), yticks = range(0, 19), \
                       title = 'Cumulative Returns of Strategies and Commodity Market Factor') # with combined strategy
plot2.get_figure().savefig("cumrets (without combined).png")

plot3 = allcumrets.plot(figsize = (15, 7), yticks = range(0, 110, 10), \
                        title = 'Cumulative Returns of Strategies and Commodity Market Factor')
plot3.get_figure().savefig("cumrets.png")



"""
PERFORMANCE STATISTICS
==============================================================================
"""
def PerformanceStats(rets, cumrets, rfr, label=None):
    
    "Calculate annualised performance statistics"
    # number of months in a year
    months = 12
    
    # pre-allocate empty dataframe for stats
    if label is None:
        stats = pd.DataFrame(index=[0])
    else:
        stats = pd.DataFrame(index=[label])         
    
    lastIndex            = cumrets.shape[0] - 1
    P_Last               = cumrets[lastIndex]            
    stats['Tot Ret']     = (P_Last - cumrets[0]) / cumrets[0]
    stats['Avg Ret']     = rets.mean()*months
    stats['rfr']         = rfr.mean()*months
    stats['SD']          = rets.std()*np.sqrt(months)
    stats['SR']          = (stats['Avg Ret'] - stats['rfr'] ) / stats['SD']
    stats['Skew']        = rets.skew()
    stats['Kurt']        = rets.kurtosis()
    stats['HWM']         = cumrets.max()
    HWM_time             = cumrets.idxmax()
    stats['HWM date']    = HWM_time.date()
                                                            
    DD                   = cumrets.cummax() - cumrets
    end_mdd              = np.argmax(DD)
    start_mdd            = np.argmax(cumrets[:end_mdd])
    
    # Maximum Drawdown defined here as POSITIVE proportional loss from peak (notation is sometimes NEGATIVE)
    stats['MDD']         = 1 - cumrets[end_mdd]/cumrets[start_mdd]       # (same as P_start - P_end) / P_start
    stats['Peak date']   = start_mdd.date()
    stats['Trough date'] = end_mdd.date()
    
    bool_P               = cumrets[end_mdd:] > cumrets[start_mdd]
    
    if (bool_P.idxmax().date() > bool_P.idxmin().date()):
        stats['Rec date']    = bool_P.idxmax().date()                                                  
        stats['MDD dur']     = (stats['Rec date'] - stats['Peak date'])[0].days
    else:
        stats['Rec date']    = stats['MDD dur']  ='Yet to recover'
    
    return stats.T         # returns transpose of pandas DF

allrets = pd.concat([EW_mthlyrets, momcumrets['EW Returns'], \
                     ivolcumrets['EW Returns'], EMAcumrets['EW Returns'], \
                     portcumrets['Returns']], axis = 1)
allrets.columns = ['Commodity Market Factor','Momentum', 'IVol', 'EMA', 'Combined']

stats = [PerformanceStats(allrets.iloc[:,0], allcumrets.iloc[:,0], mthlyRF[1:].values, "Commodity Market Factor"),
         PerformanceStats(allrets.iloc[:,1], allcumrets.iloc[:,1], mthlyRF[1:].values, "Momentum"),
         PerformanceStats(allrets.iloc[:,2], allcumrets.iloc[:,2], mthlyRF[1:].values, "IVol"),
         PerformanceStats(allrets.iloc[:,3], allcumrets.iloc[:,3], mthlyRF[1:].values, "EMA"),
         PerformanceStats(allrets.iloc[:,4], allcumrets.iloc[:,4], mthlyRF[1:].values, "Combined")]
stats = pd.concat(stats, axis = 1)

stats.to_csv('stats.csv')



"""
SENSIVITY WITH COMMODITY MARKET FACTOR
==============================================================================
"""
port_x = pd.DataFrame(allrets['Commodity Market Factor'])
port_x = sm.add_constant(port_x) # for y-intercept
port_y = pd.DataFrame(allrets['Combined'])

# linear regression of the combined strategy on the commodity market factor
port_lm = sm.OLS(port_y, port_x).fit() 
print(port_lm.summary())



"""
CONTRIBUTION BY SECTOR
==============================================================================
"""
# get commodities names for each sector
agri_liv = names[0:11] 
energy = names[11:17]
metal = names[17:]

# group portfolio returns by their sectors
portrets_bysector = pd.DataFrame(index = mth_index[13:])
portrets_bysector['Agri & Livestock'] = portrets[agri_liv].sum(axis = 1)
portrets_bysector['Energy'] = portrets[energy].sum(axis = 1)
portrets_bysector['Metals'] = portrets[metal].sum(axis = 1)
portrets_bysector['Total'] = portcumrets['Returns']

# calculate cumulative returns by sector
portcumrets_bysector = pd.DataFrame(index = mth_index[13:])
portcumrets_bysector['Agri & Livestock'] = (1 + portrets_bysector['Agri & Livestock']).cumprod()
portcumrets_bysector['Energy'] = (1 + portrets_bysector['Energy']).cumprod()
portcumrets_bysector['Metals'] = (1 + portrets_bysector['Metals']).cumprod()
# add a row of 1's at the beginning (assume $1 initial investment)
base2 = pd.DataFrame({'Agri & Livestock':[1],'Energy':[1], 'Metals':[1]}, index = mth_index[12:13])
portcumrets_bysector = pd.concat([base2, portcumrets_bysector], axis = 0) 

# calculate contribution by sector (sectoral returns/total returns)
contribution = portrets_bysector.iloc[:, 0:3].div(portrets_bysector['Total'], axis = 0)
contribution_byyear = contribution.resample('A').mean() * 100 # take yearly mean and convert to percentage form

# plot cumulative returns by sector and contribution by sector
plot4 = portcumrets_bysector.plot(figsize = (15,7), yticks = range(15), \
                          title = 'Cumulative Returns of Strategy by Sector')
plot5 = contribution_byyear.plot(figsize = (15,7), title = 'Yearly Contribution by Sector (%)')
plot5_2 = contribution_byyear.plot(figsize = (15,7), ylim = (-1000, 1000), \
                                   title = 'Yearly Contribution by Sector (%) (zoomed)')

# save plots
plot4.get_figure().savefig("sectors (cumrets).png")
plot5.get_figure().savefig("sectors (contribution).png")
plot5_2.get_figure().savefig("sectors (contribution zoomed).png")

# performance statistics by sector

sector_stats = \
[PerformanceStats(portrets_bysector['Agri & Livestock'], portcumrets_bysector['Agri & Livestock'], \
                  mthlyRF[1:].values, "Agri & Livestock"),\
PerformanceStats(portrets_bysector['Energy'], portcumrets_bysector['Energy'], \
                 mthlyRF[1:].values, "Energy"),
PerformanceStats(portrets_bysector['Metals'], portcumrets_bysector['Metals'], \
                 mthlyRF[1:].values, "Metal")]

sector_stats = pd.concat(sector_stats, axis = 1)
sector_stats.to_csv('sector_stats.csv')



"""
MOMENTUM-ADJUSTED RELATIVE STRENGTH INDEX (RSI)
==============================================================================
"""

# It should be noted that the module TA-Lib should be installed before running the RSI part
import talib
df_pct_mth = df_price_mth.pct_change()

# Storing all the positions of RSI into a dataframe
df_rsi = pd.DataFrame()
    # Set the time period as 12 for the RSI calculation
for col_index in range(0,len(df_price_mth.columns)):
    rsi = talib.RSI(df_price_mth.iloc[:,col_index], timeperiod = 12)
    df_rsi[str(col_index)] = rsi
df_rsi.columns = list(df_price_mth.columns.values)
df_rsi = df_rsi.iloc[12:,:]

position_rsi_mth = pd.DataFrame(0, index=df_rsi.index[1:], columns=df_rsi.columns)
position_arsi_mth = pd.DataFrame(0, index=df_rsi.index[1:], columns=df_rsi.columns)

# Using RSI as signal to generate positions

    # Simple RSI Strategy
for col_index in range(0,len(df_rsi.columns)):
    for row_index in range(0,len(df_rsi.index)-1):
        if df_rsi.iloc[row_index,col_index] > 70:
            position_rsi_mth.iloc[row_index,col_index] = -1
        elif df_rsi.iloc[row_index,col_index] < 30:
            position_rsi_mth.iloc[row_index,col_index] = 1
        else:
            position_rsi_mth.iloc[row_index,col_index] = 0

    # Momentum-Adjusted RSI Strategy
for col_index in range(0,len(df_rsi.columns)):
    for row_index in range(0,len(df_rsi.index)-1):
        if df_rsi.iloc[row_index,col_index] > 80:
            position_arsi_mth.iloc[row_index,col_index] = -1
        elif (df_rsi.iloc[row_index,col_index] >= 65 and df_rsi.iloc[row_index,col_index] <= 80):
            position_arsi_mth.iloc[row_index,col_index] = 0
        elif (df_rsi.iloc[row_index,col_index] >= 50 and df_rsi.iloc[row_index,col_index] <= 65):
            position_arsi_mth.iloc[row_index,col_index] = 1
        elif (df_rsi.iloc[row_index,col_index] >= 35 and df_rsi.iloc[row_index,col_index] <= 50):
            position_arsi_mth.iloc[row_index,col_index] = -1
        elif (df_rsi.iloc[row_index,col_index] >= 20 and df_rsi.iloc[row_index,col_index] <= 35):
            position_arsi_mth.iloc[row_index,col_index] = 0
        elif df_rsi.iloc[row_index,col_index] < 20:
            position_arsi_mth.iloc[row_index,col_index] = 1
        else:
            position_arsi_mth.iloc[row_index,col_index] = 0
    


# Multiply the monthly return with the positions to generate RSI strategy return
start_row = 12
df_pct_with_pos_RSI_mth = (df_pct_mth.iloc[start_row:,:])*position_rsi_mth + 1
df_pct_with_pos_RSI_mth.replace(np.nan, 0, inplace=True)
df_pct_with_pos_RSI_mth = df_pct_with_pos_RSI_mth.iloc[:len(df_pct_with_pos_RSI_mth),:]
df_pct_with_pos_aRSI_mth = (df_pct_mth.iloc[start_row:,:])*position_arsi_mth + 1
df_pct_with_pos_aRSI_mth.replace(np.nan, 0, inplace=True)
df_pct_with_pos_aRSI_mth = df_pct_with_pos_aRSI_mth.iloc[:len(df_pct_with_pos_RSI_mth),:]

# Calculate the cumulative return for the momentum-adjusted RSI strategy for each commodity
RSI_price_df_mth = pd.DataFrame(index = df_pct_with_pos_RSI_mth.index, columns = df_pct_with_pos_RSI_mth.columns)
for col_index in range(0, len(df_pct_with_pos_RSI_mth.columns)):
    RSI_price_list_mth = list()
    RSI_price_mth = 1
    for row_index in range(0, len(df_pct_with_pos_RSI_mth.index)):
        if RSI_price_mth == 0:
            RSI_price_mth = 1
        RSI_price_mth = RSI_price_mth * df_pct_with_pos_RSI_mth.iloc[row_index,col_index]
        RSI_price_list_mth.append(RSI_price_mth)   
    RSI_price_df_mth.iloc[:, col_index] = RSI_price_list_mth
RSI_price_df_mth.replace(0, np.nan, inplace=True)

aRSI_price_df_mth = pd.DataFrame(index = df_pct_with_pos_aRSI_mth.index, columns = df_pct_with_pos_aRSI_mth.columns)
for col_index in range(0, len(df_pct_with_pos_aRSI_mth.columns)):
    aRSI_price_list_mth = list()
    aRSI_price_mth = 1
    for row_index in range(0, len(df_pct_with_pos_RSI_mth.index)):
        if aRSI_price_mth == 0:
            aRSI_price_mth = 1
        aRSI_price_mth = aRSI_price_mth * df_pct_with_pos_aRSI_mth.iloc[row_index,col_index]
        aRSI_price_list_mth.append(aRSI_price_mth)   
    aRSI_price_df_mth.iloc[:, col_index] = aRSI_price_list_mth
aRSI_price_df_mth.replace(0, np.nan, inplace=True)

# Calculate the cumulative return for the RSI strtegies
RSI_ret_mth = RSI_price_df_mth.pct_change()
RSI_cum_return_mth = (RSI_ret_mth.mean(axis='columns')+1).cumprod()[1:]
RSI_cum_return_mth.index = momBS_noNA.index

aRSI_ret_mth = aRSI_price_df_mth.pct_change()
aRSI_cum_return_mth = (aRSI_ret_mth.mean(axis='columns')+1).cumprod()[1:]
aRSI_cum_return_mth.index = momBS_noNA.index


# Combining RSI strategies with other signals
    # RSI Strategy
RSIBS = position_rsi_mth
RSIBS.index = momBS_noNA.index
RSIBS_noNA = RSIBS.fillna(0)
totBS_wRSI = 2*momBS_noNA + ivolB_noNA + 2*EMABS_noNA + 2*RSIBS_noNA # weighted sum of signals
totBS_wRSI['Sum of Signals'] = totBS_wRSI.abs().sum(axis = 1) # sum of absolute signals of each row
portBS_wRSI = totBS_wRSI.iloc[:, :24].div(totBS_wRSI['Sum of Signals'], axis = 0) # weighted signals to be used as weights in portfolio

portrets_wRSI = portBS_wRSI * mthlyrets.iloc[12:] 
portcumrets_wRSI = pd.DataFrame(index = portrets_wRSI.index) 
portcumrets_wRSI['Returns'] = portrets_wRSI.sum(axis = 1) 
portcumrets_wRSI['Cumulative Returns'] = (1 + portcumrets_wRSI['Returns']).cumprod()

allcumrets_wRSI = pd.concat([RSI_cum_return_mth, \
                        portcumrets_wRSI['Cumulative Returns']], axis = 1)
allcumrets_wRSI.columns = ['RSI', 'Combined+RSI']
base_wRSI = pd.DataFrame({'RSI':[1],'Combined+RSI':[1]}, index = mth_index[12:13])
allcumrets_wRSI = pd.concat([base_wRSI, allcumrets_wRSI], axis = 0) 

    # Momentum-Adjusted RSI Strategy
aRSIBS = position_arsi_mth
aRSIBS.index = momBS_noNA.index
aRSIBS_noNA = aRSIBS.fillna(0)
totBS_waRSI = 2*momBS_noNA + ivolB_noNA + 2*EMABS_noNA + 2*aRSIBS_noNA # weighted sum of signals
totBS_waRSI['Sum of Signals'] = totBS_waRSI.abs().sum(axis = 1) # sum of absolute signals of each row
portBS_waRSI = totBS_waRSI.iloc[:, :24].div(totBS_waRSI['Sum of Signals'], axis = 0) # weighted signals to be used as weights in portfolio

portrets_waRSI = portBS_waRSI * mthlyrets.iloc[12:] 
portcumrets_waRSI = pd.DataFrame(index = portrets_waRSI.index) 
portcumrets_waRSI['Returns'] = portrets_waRSI.sum(axis = 1) 
portcumrets_waRSI['Cumulative Returns'] = (1 + portcumrets_waRSI['Returns']).cumprod()

allcumrets_waRSI = pd.concat([aRSI_cum_return_mth, \
                        portcumrets_waRSI['Cumulative Returns']], axis = 1)
allcumrets_waRSI.columns = ['Momentum-Adjusted RSI', 'Combined + MomAdj RSI']
base_waRSI = pd.DataFrame({'Momentum-Adjusted RSI':[1],'Combined + MomAdj RSI':[1]}, index = mth_index[12:13])
allcumrets_waRSI = pd.concat([base_waRSI, allcumrets_waRSI], axis = 0) 

RSI_aRSI_comparison = pd.concat([RSI_cum_return_mth,aRSI_cum_return_mth], axis = 1)
RSI_aRSI_comparison.columns = ['RSI','Mom-Adj RSI']


# Plot Graphs
plot6 = RSI_aRSI_comparison.plot(figsize = (15, 7), yticks = range(0, 2), \
                       title = 'RSI Cumulative Returns vs Mom-Adj Cumulative Returns')

plot7 = allcumrets_wRSI.plot(figsize = (15, 7), yticks = range(0, 100, 10), \
                        title = 'Cumulative Returns of Combined Strategy with RSI')

plot8 = allcumrets_waRSI.plot(figsize = (15, 7), yticks = range(0, 100, 10), \
                        title = 'Cumulative Returns of Combined Strategy with Momentum-Adjusted RSI')

plot6.get_figure().savefig("RSI cumrets.png")
plot7.get_figure().savefig("cumrets (with RSI).png")
plot8.get_figure().savefig("cumrets (with Mom-Adj RSI).png")

# Summary Stats of the RSI Strategies

RSI_stats = \
[PerformanceStats(portcumrets_wRSI['Returns'], portcumrets_wRSI['Cumulative Returns'], \
                  mthlyRF[1:].values, "Cumulative Returns with RSI"),\
PerformanceStats(portcumrets_waRSI['Returns'] , portcumrets_waRSI['Cumulative Returns'], \
                 mthlyRF[1:].values, "Cumulative Returns with MomAdj-RSI")]

RSI_stats = pd.concat(RSI_stats, axis = 1)
RSI_stats.to_csv('RSI_stats.csv')



end_time = time.time()
print(str(end_time - start_time) + " seconds taken")
