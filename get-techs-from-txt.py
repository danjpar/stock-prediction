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

def getTechnicals(values, smas):
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

def getStockData(filename):
    ''' file input as 7 columns separated by commas, rows as new lines '''
    f = open(filename, 'r')
    f = f.read()
    splitfile = f.split('\n')
    stockfile = splitfile[:len(splitfile)-2]
    
    readfile = pd.DataFrame([sub.split(",") for sub in stockfile])
    readfile.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    
    datef = readfile['date']
    timef = readfile['time']
    
    datef = pd.DataFrame(datef.str.split('/',2).tolist(), columns = ['month','day','year'])
    datef['year'] = datef['year'].str[2:]
    datef = datef.astype('uint8')
    timef = pd.DataFrame(timef.str.split(':',1).tolist(), columns = ['hour','minute'])
    timef = timef.astype('uint8')
    volumes = readfile['volume'].astype('uint32')
    prices = (readfile.drop(['date', 'time' ,'volume'], axis=1)).astype('float64')
    
    data = pd.concat([datef, timef, prices,  volumes], axis=1)
    
    data = data.loc[data[(data['hour'] == 9) & (data['minute'] == 30)].index[0]:] # start on first opening minute.
    year = data['year'] + 10
    year[year > 100] = year - 100
    data['year'] = year
    #ah1 = data[((data.hour == 9) & (data.minute < 30)) | (data.hour < 9)]
    #ah2 = data[((data.hour == 16) & (data.minute > 4)) | (data.hour > 16)]
    nh = data[((data.hour == 16) & (data.minute <= 4)) | ((data.hour == 9) & (data.minute >= 30)) | ((data.hour > 9) & (data.hour < 16))]
    return nh

def newTimeframe(nh, interval, tick):
    start = 0
    maxtick = 0
    timearray = ['year', 'month', 'day']
    if interval == 'minute':
        maxtick = 60-tick
        timearray.extend(['hour', 'minute'])
    if interval == 'hour':
        start = 9
        maxtick = start+8-tick
        timearray.extend(['hour'])

    newdf = pd.DataFrame()
    while start <= maxtick:
        end = tick + start
        if interval == 'minute':
            nhtimeslices = nh[(nh['minute'] >= start)&(nh['minute'] < end)].groupby(['year', 'month', 'day', 'hour'], as_index=False)
            newtimes = pd.DataFrame(nhtimeslices.minute.min())
        if interval == 'hour':
            nhtimeslices = nh[((nh['hour'] == start)&(nh['minute'] >= 30))|((nh['hour'] == end)&(nh['minute'] < 30))|((nh['hour'] > start)&(nh['hour'] < end))].groupby(['year', 'month', 'day'], as_index=False)
            newtimes = pd.DataFrame(nhtimeslices.hour.min())
        if interval == 'day':
            nhtimeslices = nh.groupby(['year', 'month', 'day'], as_index=False)
            newtimes = pd.DataFrame(nhtimeslices.hour.min())
        
        newlows = pd.DataFrame(nhtimeslices.low.min()).low
        newhighs = pd.DataFrame(nhtimeslices.high.max()).high
        newvols = pd.DataFrame(nhtimeslices.volume.sum()).volume
        newopens = pd.DataFrame(nhtimeslices.open.first()).open
        newcloses = pd.DataFrame(nhtimeslices.close.last()).close
        
        if interval == 'minute':
            newtimes['minute'] = start
        if interval == 'hour':
            newtimes['minute'] = 30
        if interval == 'day':
            newtimes = newtimes.drop(['hour'], axis=1)
    
        newnewdf = pd.concat([newtimes, newopens, newhighs, newlows, newcloses, newvols], axis=1)
        newdf = pd.concat([newdf, newnewdf], axis=0)
        start += tick
    
    newdf = newdf.set_index(timearray).sort_index()
    newdf = newdf.reset_index(level=timearray)
    
    newdf['change'] = newdf['close'] - newdf['open']
    newdf['percent'] = (newdf['change']/newdf['open'])*100
    newdf = newdf.round(3)
    smas = [10, 20, 50, 200]
    smas_data, extended_data, start_point = getTechnicals(newdf['close'], smas)
    newdf = newdf[start_point:]
    newdf = newdf.reset_index(drop=True)
    file_out = 'IBM/IBM_adj-'+str(tick)+interval+'-smas.csv'
    new_exData = pd.concat([newdf, extended_data, smas_data], axis=1)
    new_exData.to_csv(file_out, index=False)

nh = getStockData('IBM_adjusted.txt')
newTimeframe(nh, 'day', 1) 
ticks = [1, 2, 4]
for tick in ticks:
    newTimeframe(nh, 'hour', tick)
ticks = [5, 15, 30]
for tick in ticks:
    newTimeframe(nh, 'minute', tick)
