import pandas as pd
import numpy as np
import time
import requests

def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas # as a numpy array


def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def computeMACD(x, slow=26, fast=12):
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emafast - emaslow

def getLongestSMA(smas):
    longestSMA = 0
    for sma in smas:
        if sma > longestSMA:
            longestSMA = sma
    return longestSMA

def getTechnicals(values, smas):
    longestSMA = getLongestSMA(smas)
    start_point = longestSMA - 1
    smas_data = pd.DataFrame()
    for sma in smas:
        sma_data = movingaverage(values, sma)
        if sma != longestSMA:
            sma_data = sma_data[longestSMA - sma:]
        smas_data['SMA'+str(sma)] = sma_data
        
    macd = computeMACD(values)[start_point:]
    macd_ema9 = ExpMovingAverage(macd, 9)
    ema30 = ExpMovingAverage(values, 30)[start_point:]
    rsi = rsiFunc(values)[start_point:]
    extended_data = pd.DataFrame({'RSI': rsi, 'MACD': macd,'MACD-EMA9': macd_ema9, 'EMA30': ema30})
    extended_data = extended_data.round(5)
    smas_data = smas_data.round(3)
    return smas_data, extended_data, start_point

nowtime = int(time.time())
starttime = str(nowtime - 5097600) # intraday data limited to 60 days for 15min timeframe
endtime = str(nowtime)

ticks = ['2m', '5m', '15m', '60m']
smas = [10, 20, 50, 200]
stock = 'SPY'
for tick in ticks:
    urltovisit = 'https://query2.finance.yahoo.com/v8/finance/chart/'+stock+'?symbol='+stock+'&period1='+starttime+'&period2='+endtime+'&interval='+tick
    req = requests.get(urltovisit)
    timestamps = pd.DataFrame.from_dict(req.json()['chart']['result'][0]['timestamp'])
    timestamps.columns = ['date']
    data = pd.DataFrame.from_dict(req.json()['chart']['result'][0]['indicators']['quote'][0])
    data = pd.concat([timestamps, data], axis=1)
    data = data.dropna(axis=0, subset=['close'], how='any')
    smas_data, extended_data, start_point = getTechnicals(data['close'], smas)
    fileoutput = stock+'-'+tick+'.csv'
    data['change'] = (data['close'].diff(periods=1)).fillna(0)
    data['percent'] = (data['change']/(data['close']-data['change']))*100
    data = data[start_point:]
    data = data.reset_index(drop=True)
    data = data.round(3)
    new_exdata = pd.concat([data, extended_data, smas_data], axis=1)
    new_exdata.to_csv(fileoutput, index=False)
