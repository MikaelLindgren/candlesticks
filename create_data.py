
# Raw Package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Data Source
import yfinance as yf
import math
#Data viz
import plotly.graph_objs as go
import finance_functions as ff
#Interval required 1 minute


df = pd.read_csv("nasdaq_screener_1636808568043.csv")
symbol_dictionary = {}
sym_list = list(df["Symbol"])
name_list = list(df["Name"])
vol_list = list(df["Volume"])
vol_list = np.asarray(vol_list)
indexes = np.where(vol_list > 10000000)
indexes = indexes[0]
good_syms = []
for index in indexes:
    good_syms.append(sym_list[int(index)])

num_errors = 0 
all_data = {}
faulty_sumbols = []
for symbol in good_syms:
    data = yf.download(tickers=symbol, start="2021-11-08", end="2021-11-09", interval='1m')
    try:
        all_data[symbol] = ff.extract_training(data,delay=5, win_size=10)
    except:
        faulty_sumbols.append(symbol)
df = pd.DataFrame()

for k in all_data.keys():
    temp = pd.DataFrame()
    for j in all_data[k].keys():
        row = all_data[k][j]["candle_pic"]
        row = np.append(row,np.asarray(all_data[k][j]["label"]) )
        row = np.append(row, np.asarray(all_data[k][j]["vol"]))
        row = np.reshape(row,(1,-1))
        row = pd.DataFrame(row)
        temp = temp.append(row)
    df = df.append(temp)
print(df)
df.to_csv("finance_data.csv")