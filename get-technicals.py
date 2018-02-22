import numpy as np
import pandas as pd

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


def computeMACD(x, slow, fast):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow


def printTA (data, MA1, MA2, MA3, MA4):
    closep = data['Close']
    openp = data['Open']
    highp = data['High']
    lowp = data['Low']
    volp = data['Volume']
    
    Aves1 = movingaverage(closep, MA1)
    Aves2 = movingaverage(closep, MA2)
    Aves3 = movingaverage(closep, MA3)
    Aves4 = movingaverage(closep, MA4)
    rsi = rsiFunc(closep)
    emaslow, emafast, macd = computeMACD(closep, 30, 10)
    MACDema9 = ExpMovingAverage(macd, 9)
    
    MACDema9 = MACDema9[MA4-1:]
    macd = macd[MA4-1:]
    emafast = emafast[MA4-1:]
    emaslow = emaslow[MA4-1:]
    rsi = rsi[MA4-1:]
    Aves1 = Aves1[MA4-MA1:]
    Aves2 = Aves2[MA4-MA2:]
    Aves3 = Aves3[MA4-MA3:]
    newDate = data['Date']
    newDate = newDate[MA4-1:]

    closep = closep[MA4-1:]
    openp = openp[MA4-1:]
    highp = highp[MA4-1:]
    lowp = lowp[MA4-1:]
    volp = volp[MA4-1:]

    extended_data = pd.DataFrame({'Date': newDate, 'Open': openp, 'Close': closep, 'High': highp, 'Low': lowp, 'Volume': volp, 'SMA1': Aves1, 'SMA2': Aves2, 'SMA3': Aves3, 'SMA4': Aves4, 'RSI': rsi, 'MACD': macd, 'MACD-EMA9': MACDema9})
    extended_data.to_csv('new-data.csv', columns=['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'SMA1', 'SMA2', 'SMA3', 'SMA4', 'RSI', 'MACD', 'MACD-EMA9'], index=False)


x = pd.read_csv('daily-SPY.csv')
x.drop(['Adj Close'], axis=1)
printTA(x, 10, 20, 50, 200)