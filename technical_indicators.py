# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:50:56 2016

@author: Administrator
"""

import pandas as pd
import numpy as np

# Moving Average
def ma(price,n):
    p = price.rolling(center = False, window = n).mean()
    return p.fillna(0)

# Expoenetially Weighted Moving Average (alpha = 4, span = 9 )
def ewma(price, n):
    return price.ewm(span = n, min_periods = 0, ignore_na = False, adjust = True).mean()

######## MACD, Signal and Histogram #############
#  Moving Average Convergence/Divergence (MACD)
def MACD(price, n1, n2):
    return ewma(price, n1) - ewma(price, n2)

# Signal
def signal(price, n1, n2, span):
    p = MACD(price,n1,n2)
    return ewma(p,span)

# Histogram
def histogram(price, n1, n2, span):
    return MACD(price, n1, n2) - signal(price, n1, n2, span)

# RSI (Relative Strength Index)
def rsi(price, n):
    gain = price.diff()
    gain[np.isnan(gain)] = 0
    up, down = gain.copy(), gain.copy()
    up[up<0] = 0
    down[down>0] = 0
    upEWMA = ewma(up,n)
    downEWMA = ewma(down.abs(),n)
    rs = upEWMA/downEWMA
    return (100 - (100/(1+rs))).fillna(0)

# Stochastics (%K)
def STOK(price,high,low,n,d):
    highp = high.rolling(window = n).max()
    lowp = low.rolling(window = n).min()
    stok = 100*(price - lowp)/(highp-lowp)
    return stok.fillna(0),ma(stok,d).fillna(0), ma(ma(stok,d),d).fillna(0)

# Momentum and Rate of Cahnge(7 day)
def Momentum(price,n):
    p = price.diff(n)
    return p.fillna(0)

def RateOfChange(price,n):
    p = price.diff(n)/price.shift(n)
    return p.fillna(0)

# Moving variance
def MovingVariance(price,n):
    p = price.rolling(window = n).var()
    return p.fillna(0)

#Commodity Channel Index (CCI)iance
def CCI(price, high, low, n, d):
    tp = (price + high + low)/3
    MeanDev = MeanDeviation(tp,d)
    CCI = (tp - ma(tp,n))/(.015 * MeanDev)
    return CCI.fillna(0)

def MeanDeviation(price, d):
    mad = price.rolling(window =d, center = False).apply(lambda x: np.abs(x - x.mean()).mean())
    return mad.fillna(0)

# Chaikin Oscillator
def Chaikin(price, high, low, volume):
    MoneyFlowMultiplier = ((price - low) - (high - low))/ (high - low)
    MoneyFlowVolume = MoneyFlowMultiplier * volume
    ADL = price.copy()
    ADL[0] = 0
    for i in range(1,len(price)):
        ADL[i] = ADL[i-1] + MoneyFlowVolume[i] 
    Chaikin = ewma(ADL,3) - ewma(ADL,10)
    return Chaikin.fillna(0)

# Disparity Index (10)
def DisparityIndex(price,n) :
    DI = 100*(price - ma(price,n))/ma(price,n)
    DI = DI.replace(np.inf,np.nan)
    return DI.fillna(0)

# Williams %R
def WilliamR(price,high,low,n):
    highp = high.rolling(window = n).max()
    lowp = low.rolling(window = n).min()
    WR = (highp - price) / (highp - lowp)
    return WR.fillna(0)

# Volatility
def volatility(price,n):
    volatility = price.rolling(window = n).std()/ma(price,n)
    return volatility.fillna(0)

# Standard deviation
def stdev(price,n):
    stdev = price.rolling(window = n, center = False).std()
    return stdev.fillna(0)