import pandas as pd
import numpy as np

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
    return emaslow, emafast, emafast - emaslow

def getLongestSMA(smas):
    longestSMA = 0
    for i in smas:
        if i > longestSMA:
            longestSMA = i
    return longestSMA

def gettechnicals(values, smas):
    longestSMA = getLongestSMA(smas)
    start_point = longestSMA - 1
    smas_data = pd.DataFrame()
    for sma in smas:
        sma_data = movingaverage(values, sma)
        if sma != longestSMA:
            sma_data = sma_data[longestSMA - sma:]
        smas_data['SMA'+str(sma)] = sma_data
    
    emaslow, emafast, macd = computeMACD(values)
    macd = macd[start_point:]
    macd_ema9 = ExpMovingAverage(macd, 9)
    ema30 = (ExpMovingAverage(values, 30))[start_point:]
    rsi = (rsiFunc(values))[start_point:]
    extended_data = pd.DataFrame({'RSI': rsi, 'MACD': macd,'MACD-EMA9': macd_ema9, 'EMA30': ema30})
    extended_data = extended_data.round(5)
    smas_data = smas_data.round(3)
    return smas_data, extended_data, start_point

def getstockdata(file_in):
    f = open(file_in, 'r').read()
    stockfile = []
    splitfile = f.split('\n')
    
    for eachline in splitfile:
        splitline = eachline.split(',')
        if len(splitline)==7:
            stockfile.append(eachline)
    
    readfile = pd.DataFrame([sub.split(",") for sub in stockfile])
    readfile.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    
    datef = pd.DataFrame(readfile['date'].str.split('/',2).tolist(), columns = ['month','day','year'])
    datef['year'] = datef['year'].str[2:]
    datef = datef.astype('uint8')
    timef = pd.DataFrame(readfile['time'].str.split(':',1).tolist(), columns = ['hour','minute'])
    timef = timef.astype('uint8')
    volumes = readfile['volume'].astype('uint32')
    prices = (readfile.drop(['date', 'time' ,'volume'], axis=1)).astype('float64')
    
    new_data = pd.concat([datef, timef, prices,  volumes], axis=1)
    
    new_data['change'] = (new_data['close'].diff(periods=1)).fillna(0)
    data['percent'] = (data['change']/(data['close']-data['change']))*100
    
    return new_data

file_in = 'IBM_adjusted.txt'
''' file input as 7 columns separated by commas, rows as new lines '''
file_out = 'IBM/IBM-SMAS.csv'
data = getstockdata(file_in)
smas = [10, 20, 50, 200]
smas_data, extended_data, start_point = gettechnicals(data['close'], smas)
data = data[start_point:]
data = data.reset_index(drop=True)
data = data.round(3)
year = data['year'] + 10
year[year > 100] = year - 100
data['year'] = year
new_exdata = pd.concat([data, extended_data, smas_data], axis=1)
#new_exdata.to_csv(file_out, index=False)
