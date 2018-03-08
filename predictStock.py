import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import requests
import time
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
###
def getStockData(filename): ### for txt files with columns date, time, open, high, low, close, volume
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
    data['change'] = data['close'] - data['open']
    data['percent'] = (data['change']/data['open'])*100
    return data
###
def getDataFromYahoo(stock, interval, tick):
    nowtime = int(time.time())
    if (interval == 'minute')|(interval == 'hour'):
        starttime = str(nowtime - 5097600) # intraday data limited to 60 days for 15min timeframe
    else:
        starttime = str(0)  
    endtime = str(nowtime)
    dtick = str(tick)+interval[0]
    urltovisit = 'https://query2.finance.yahoo.com/v8/finance/chart/'+stock+'?symbol='+stock+'&period1='+starttime+'&period2='+endtime+'&interval='+dtick
    req = requests.get(urltovisit)
    timestamps = pd.DataFrame.from_dict(req.json()['chart']['result'][0]['timestamp'])
    timestamps = pd.to_datetime(timestamps[0], unit='s')
    newdates = pd.DataFrame({'date': timestamps})
    newdates['new_date'] = [d.date() for d in newdates.date]
    newdates['time'] = [d.time() for d in newdates.date]
    newdates = newdates.astype('str')
    datef = pd.DataFrame(newdates.new_date.str.split('-',2).tolist(), columns = ['year','month','day'])
    datef['year'] = datef['year'].str[2:]
    datef = datef.astype('uint8')
    timef = pd.DataFrame(newdates.time.str.split(':',1).tolist(), columns = ['hour','minute'])
    timef['minute'] = timef['minute'].str[:2]
    timef = timef.astype('uint8')
    data = pd.DataFrame.from_dict(req.json()['chart']['result'][0]['indicators']['quote'][0])
    cols = data.columns.tolist()
    cols = [cols[-2]]+cols[1:3]+[cols[0]]+[cols[-1]]
    data = data[cols]
    data = pd.concat([datef, timef, data], axis=1)
    data = data.dropna(axis=0, subset=['close'], how='any')
    data['change'] = data['close'] - data['open']
    data['percent'] = (data['change']/data['open'])*100   
    data.hour = data.hour - 5
    return data
###
def newTimeframe(ogdata, interval, tick, includeah): ### convert time frames
    data = ogdata.copy()
    start = 0
    maxtick = 0
    datearray = ['year', 'month', 'day']
    if includeah != True:
        data = data[((data.hour == 16) & (data.minute <= 4)) | ((data.hour == 9) & (data.minute >= 30)) | ((data.hour > 9) & (data.hour < 16))]
    if interval == 'minute':
        maxtick = 60-tick
        datearray.extend(['hour', 'minute'])
    elif interval == 'hour':
        maxtick = 24-tick
        datearray.extend(['hour'])
    newdf = pd.DataFrame()
    while start <= maxtick:
        end = tick + start
        if interval == 'minute':
            datatimeslices = data[(data['minute'] >= start)&(data['minute'] < end)].groupby(['year', 'month', 'day', 'hour'], as_index=False)
            newtimes = datatimeslices.minute.min()
            newtimes['minute'] = start
        if interval == 'hour':
            if includeah != True:
                datatimeslices = data[((data['hour'] == start)&(data['minute'] >= 30))|((data['hour'] == end)&(data['minute'] < 30))|((data['hour'] > start)&(data['hour'] < end))].groupby(['year', 'month', 'day'], as_index=False)
                newtimes['minute'] = 30
            else:
                datatimeslices = data[((data['hour'] >= start)&(data['hour'] < end))].groupby(['year', 'month', 'day'], as_index=False)
                newtimes['minute'] = 0
            newtimes = datatimeslices.hour.min()
        if interval == 'day':
            datatimeslices = data.groupby(datearray, as_index=False)
            newtimes = datatimeslices.hour.min()
            newtimes = newtimes.drop(['hour'], axis=1)
        newlows = datatimeslices.low.min().low
        newhighs = datatimeslices.high.max().high
        newvols = datatimeslices.volume.sum().volume
        newopens = datatimeslices.open.first().open
        newcloses = datatimeslices.close.last().close
        newnewdf = pd.concat([newtimes, newopens, newhighs, newlows, newcloses, newvols], axis=1)
        newdf = pd.concat([newdf, newnewdf], axis=0)
        start += tick
    ###
    newdf = newdf.set_index(datearray).sort_index()
    newdf = newdf.reset_index(level=datearray)
    newdf['change'] = newdf['close'] - newdf['open']
    newdf['percent'] = (newdf['change']/newdf['open'])*100
    new_exData = getTechnicals(newdf)
    return new_exData
###
def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)
    for i in range(n, len(prices)):
        delta = deltas[i-1]
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
###
def movingaverage(values,window):
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(values, weights, 'valid')
    return smas
###
def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a
###
def computeMACD(x, slow=26, fast=12):
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow
###
def getStochastics(close, low, high, n): 
    STOK = ((close - low.rolling(window=n).min()) / (high.rolling(window=n).max() - low.rolling(window=n).min())) * 100
    STOD = STOK.rolling(window=3).mean()
    return STOK, STOD
###
def getLongestSMA(smas):
    longestSMA = 0
    for i in smas:
        if i > longestSMA:
            longestSMA = i
    return longestSMA
###
def getTechnicals(ogdata):
    data = ogdata.copy()
    values = data.close
    smas = [10, 20, 30]
    longestSMA = getLongestSMA(smas)
    start_point = longestSMA - 1
    smas_data = pd.DataFrame()
    for sma in smas:
        sma_data = movingaverage(values, sma)
        if sma != longestSMA:
            sma_data = sma_data[longestSMA - sma:]
        smas_data['SMA'+str(sma)] = sma_data
    
    stoK, stoD = getStochastics(values, data.low, data.high, 14)
    stoK = stoK[start_point:].round(1)
    stoD = stoD[start_point:].round(1)
    emaslow, emafast, macd = computeMACD(values)
    macd = macd[start_point:]
    macd_ema9 = ExpMovingAverage(macd, 9)
    ema30 = (ExpMovingAverage(values, 30))[start_point:]
    rsi = (rsiFunc(values))[start_point:].round(1)
    extended_data = pd.DataFrame({'RSI': rsi, 'MACD': macd,'MACD-EMA9': macd_ema9, 'EMA30': ema30, 'stoK': stoK, 'stoD': stoD})
    extended_data = extended_data.reset_index(drop=True)
    data = data[start_point:].reset_index(drop=True)
    data = pd.concat([data, extended_data, smas_data], axis=1)
    return data.round(3)
###
def getPrediction(data, tick):
    df = data.copy()
    df = getTechnicals(df)
    timeframe = tick
    newdf = df.tail(1).copy()
    new_index = newdf.index + 1
    if 0 in df.minute.values:
        newdf.minute = newdf.minute + timeframe
    elif 10 in df.hour.values:
        newdf.hour = newdf.hour + timeframe
    else:
        newdf.day = newdf.day + timeframe
    newdf.index = new_index
    if (17 not in df.hour.values)|(8 not in df.hour.values):
        if newdf.iloc[0].minute >= 60:
            newdf.hour += 1
            newdf.minute = 60 - newdf.iloc[0].minute
        if (newdf.iloc[0].hour == 16)&(newdf.iloc[0].minute >= 5):
            newdf.hour = 9
            newdf.minute = 30
    newdf = newdf.drop(newdf.columns[6:], axis=1)
    forecast_cols = ['open', 'high', 'low', 'close', 'volume']
    #print('Confidence:')
    for forecast_col in forecast_cols:
        df.fillna(value=-37373, inplace=True)
        forecast_out = 1
        df['label'] = df[forecast_col].shift(-forecast_out)
        if 10 in df.hour.values:
            X=np.array(df.drop(['label','year','month','day'], axis=1))
        else:
            X=np.array(df.drop(['label','year','month'], axis=1))

        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]
        df.dropna(inplace=True)
        y = np.array(df['label'])
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        print(mean_absolute_error(y_test, clf.predict(X_test)))
        #confidence = clf.score(X_test, y_test)
        #print(forecast_col + ': ' + str(confidence.round(4)*100) + '%')
        forecast_set = clf.predict(X_lately)
        newdf[forecast_col] = forecast_set
        df['next-'+forecast_col] = df['label']

    newdf['change'] = (newdf['close'] - newdf['open']).round(3)
    newdf['percent'] = (newdf['change']/newdf['open']*100).round(3)
    newdf = newdf.round(3)
    newdf.volume = int(newdf.volume.round())
    df = data.drop(data.columns[12:], axis=1)
    df = pd.concat([df, newdf], axis=0)
    df = getTechnicals(df.tail(40)) #must be larger than longestSMA + macd-ema
    newdf = df.tail(1)
    newdf.index = new_index
    return newdf
###

#original_data = getStockData('IBM_adjusted.txt')
#data = original_data.copy()
#newdata = newTimeframe(data, 'minute', 15, False)
data = getDataFromYahoo('spy', 'hour', 1)
vardata = getTechnicals(data.copy())
#vardata = newdata.copy()
i = 0
x = 5
while i < x:
    i += 1
    vardata = pd.concat([vardata, getPrediction(vardata, 1)], axis=0)
    
#print(vardata.tail(x))
    
#ah1 = data[((data.hour == 9) & (data.minute < 30)) | (data.hour < 9)]
#ah2 = data[((data.hour == 16) & (data.minute > 4)) | (data.hour > 16)]
#nh = data[((data.hour == 16) & (data.minute <= 4)) | ((data.hour == 9) & (data.minute >= 30)) | ((data.hour > 9) & (data.hour < 16))]