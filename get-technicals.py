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
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow

def printTA (data, smas, timeFrame):
    closep = data['close']
    openp = data['open']
    highp = data['high']
    lowp = data['low']
    volp = data['volume']
    newDate = data['date']
    
    longestSMA = 0
    smas_data = pd.DataFrame()
        
    for i in smas:
        if i > longestSMA:
            longestSMA = i
        
    for x in smas:
        sma_data = movingaverage(closep, x)
        if x != longestSMA:
            sma_data = sma_data[longestSMA - x:]
        sma_df = pd.DataFrame(sma_data)
        sma_df.columns = ['SMA'+str(x)]
        smas_data = pd.concat([smas_data, sma_df], axis=1)
    
    rsi = rsiFunc(closep)
    emaslow, emafast, macd = computeMACD(closep, 26, 13)
    MACDema9 = ExpMovingAverage(macd, 9)
    
    start_point = longestSMA - 1
    
    MACDema9 = MACDema9[start_point:]
    macd = macd[start_point:]
    emafast = emafast[start_point:]
    emaslow = emaslow[start_point:]
    rsi = rsi[start_point:]
    newDate = newDate[start_point:]
    
    new_closep = closep[start_point:]
    new_openp = openp[start_point:]
    new_highp = highp[start_point:]
    new_lowp = lowp[start_point:]
    new_volp = volp[start_point:]
    
    fileOutput = stock+'/'+stock+'-'+timeFrame+'.csv'
    
    extended_data = pd.DataFrame({'Date': newDate, 'Open': new_openp,
                                  'Close': new_closep, 'High': new_highp,
                                  'Low': new_lowp, 'Volume': new_volp,
                                  'RSI': rsi, 'MACD': macd,
                                  'MACD-EMA9': MACDema9}, columns=['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'MACD-EMA9'])
    
    extended_data = extended_data.reset_index(drop=True)
    new_exData = pd.concat([extended_data, smas_data], axis=1)
    new_exData.to_csv(fileOutput, index=False)


nowTime = int(time.time())
startTime = str(nowTime - 5097600) # intraday data limited to 60 days for 15min timeframe
endTime = str(nowTime)

ticks = ['2m', '5m', '15m', '60m']
smas = [10, 20, 50, 200]
stock = 'SPY'
for i in ticks:
    urlToVisit = 'https://query2.finance.yahoo.com/v8/finance/chart/'+stock+'?symbol='+stock+'&period1='+startTime+'&period2='+endTime+'&interval='+i
    req = requests.get(urlToVisit)
    timestamps = pd.DataFrame.from_dict(req.json()['chart']['result'][0]['timestamp'])
    timestamps.columns = ['date']
    new_data = pd.DataFrame.from_dict(req.json()['chart']['result'][0]['indicators']['quote'][0])
    new_data = pd.concat([timestamps, new_data], axis=1)
    new_data = new_data.dropna(axis=0, subset=['close'], how='any')
    printTA(new_data, smas, i)
