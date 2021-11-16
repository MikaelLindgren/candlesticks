
# Raw Package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Data Source
import yfinance as yf
import math
#Data viz
import plotly.graph_objs as go

#Interval required 1 minute
# GOAL IS TO FIND SUITABLE INDEX FOR FINDING TRAINING DATA
#data = yf.download(tickers='AAPL', start="2021-11-09", end="2021-11-10", interval='1m')
def falta(data,operator):
    ret_val = np.convolve(operator,data)
    ret_val = ret_val[len(operator)-1:-(len(operator)-1)]
    return ret_val
def falta3(data,operator):
    ret_val = np.convolve(data, operator)
    ret_val = ret_val[len(operator)-1:-(len(operator)-1)]
    return ret_val
def index_getter(data, threshhold):
    if threshhold > 0:
        index = np.where(data > threshhold)
    if threshhold < 0:
        index = np.where(data < threshhold)
    index = list(index)
    index = index[0]
    return index
def get_ones(num):
    ret_val = []
    for i in range(num):
        ret_val.append(1)
    return ret_val
def otoc(data, index):
    op = data["Open"][index]
    cl = data["Close"][index]
    diff = cl-op
    rel_diff = diff / op
    return rel_diff
def htol(data, index):
    high = data["High"][index]
    low = data["Low"][index]
    diff = low-high
    rel_diff = diff / ((high+low)/2)
    return rel_diff
def vol(data, index):
    return data["Volume"][index]
def zeroish(value):
    if abs(value) < 0.05:
        return True
    return False
def upper_wick(data, index):
    high = data["High"][index]
    op = data["Open"][index]
    cl = data["Close"][index]
    over = high-max(op,cl)
    body = op-cl
    # if zeroish(body):
    #     return over / 100 
    up_wick = over / max(op,cl)
    return up_wick
def lower_wick(data, index):
    low = data["Low"][index]
    op = data["Open"][index]
    cl = data["Close"][index]
    under = min(op,cl)-low
    body = op-cl
    # if zeroish(body):
    #     return under / 100
    low_wick = under / max(op,cl)
    return low_wick

def candle_sticker(data,index):
    return [otoc(data,index),htol(data,index), upper_wick(data, index),lower_wick(data, index)]

def candle_picture(data,start, fin):
    candle_pic = []
    for i in range(start,fin):
        candle_pic.append(candle_sticker(data,i))
    return candle_pic

def paint(ridge,close, index_plus,index_minus,delay, rel_close,data, falta2, threshhold):
    print(len(ridge))
    print(close)
    print(data.columns)
    len(close)
    plt.figure()
    plt.subplot(411)
    plt.plot(close)
    plt.plot(index_plus-delay, close[index_plus-delay],'o')
    plt.plot(index_minus-delay, close[index_minus-delay],'o')
    plt.subplot(412)
    plt.plot(rel_close)
    plt.subplot(413)
    plt.plot(ridge)
    plt.hlines(y = 0, xmin = 0 , xmax = len(ridge), color = "r")
    plt.hlines(y = np.mean(ridge), xmin = 0 , xmax = len(ridge), color = "y")
    plt.subplot(414)
    plt.plot(falta2)
    plt.hlines(y=threshhold, xmin=0, xmax = len(falta2))
    plt.hlines(y=-threshhold, xmin = 0, xmax = len(falta2))
    plt.plot(index_plus-delay, falta2[index_plus-delay],'o')
    plt.plot(index_minus-delay, falta2[index_minus-delay],'o')
    print(data["Volume"])
    plt.show()
    print(len(falta2))
    print(len(ridge))
    print(len(close))


def extract_training(data, delay, win_size):
    close = data["Close"]
    close = np.array(close)
    rel_close = close / max(close)
    ridge = falta(rel_close, [1,-1])

    close = close[1:]
    data = data[1:]
    rel_close = rel_close[1:]
    falta2 = falta(ridge, get_ones(delay))
    ridge = ridge[delay-1:]
    rel_close = rel_close[delay-1:]
    close = close[delay-1:]
    data = data[delay-1:]

    #index_plus = index_getter(falta2,threshhold)# -delay
    #index_minus = index_getter(falta2, -threshhold)# - delay
    #paint(ridge,close, index_plus,index_minus,delay, rel_close,data, falta2, threshhold)
    packets = {}
    for i in range(20-delay,len(close)):
        packets[i] = {}
        packets[i]["candle_pic"] = candle_picture(data,i-win_size-delay,i-delay)
        packets[i]["vol"] = vol(data,i-delay)
        packets[i]["label"] = falta2[i] #"neutral"
        # if i in index_plus:
        #     packets[i]["label"] = "plus"
        # if i in index_minus:
        #     packets[i]["label"] = "minus"
    packets = format_packets(packets, win_size)

    return packets
#packets = extract_training(data, 5, .015, win_size )
def list_list_to_nparray(list_list, win_size):
    x = np.hstack(list_list)
    x = np.reshape(x,(1,-1))
    #print(type(x))
    #print(len(x))
    #print(x.size)
    # for i in range(10):
    #     for j in range(2):
    #         x[i][j] = x[i][j]*.5
    return x
def format_packets(packets, win_size):
    for k in packets.keys():
        x = list_list_to_nparray(packets[k]["candle_pic"], win_size)
        packets[k]["candle_pic"] = x
    return packets
