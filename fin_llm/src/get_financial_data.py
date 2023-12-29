import os
import re
import csv
import math
import time
import json
import random
import finnhub
import datasets
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import defaultdict
from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm


def bin_mapping(ret):
    
    up_down = 'U' if ret >= 0 else 'D'
    integer = math.ceil(abs(100 * ret))
    
    return up_down + (str(integer) if integer <= 5 else '5+')


def get_returns(START_DATE, END_DATE, stock_symbol):
    
    # Download historical stock data
    stock_data = yf.download(stock_symbol, start=START_DATE, end=END_DATE)
    
    weekly_data = stock_data['Adj Close'].resample('W').ffill()
    weekly_returns = weekly_data.pct_change()[1:]
    weekly_start_prices = weekly_data[:-1]
    weekly_end_prices = weekly_data[1:]

    weekly_data = pd.DataFrame({
        'Start Date': weekly_start_prices.index,
        'Start Price': weekly_start_prices.values,
        'End Date': weekly_end_prices.index,
        'End Price': weekly_end_prices.values,
        'Weekly Returns': weekly_returns.values
    })
    
    weekly_data['Bin Label'] = weekly_data['Weekly Returns'].map(bin_mapping)

    return weekly_data


def get_news(symbol, data, finnhub_client):
    
    news_list = []
    
    for end_date, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        print(symbol, ': ', start_date, ' - ', end_date)
        time.sleep(1) # control qpm
        weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        weekly_news = [
            {
                "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                "headline": n['headline'],
                "summary": n['summary'],
            } for n in weekly_news
        ]
        weekly_news.sort(key=lambda x: x['date'])
        news_list.append(json.dumps(weekly_news))
    
    data['News'] = news_list
    
    return data


def get_basics(START_DATE, symbol, data, finnhub_client,  always=False):
    
    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
    
    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    
    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)
        
    basic_list.sort(key=lambda x: x['period'])
            
    for i, row in data.iterrows():
        
        start_date = row['End Date'].strftime('%Y-%m-%d')
        last_start_date = START_DATE if i < 2 else data.loc[i-2, 'Start Date'].strftime('%Y-%m-%d')
        
        used_basic = {}
        for basic in basic_list[::-1]:
            if (always and basic['period'] < start_date) or (last_start_date <= basic['period'] < start_date):
                used_basic = basic
                break
        final_basics.append(json.dumps(used_basic))
        
    data['Basics'] = final_basics
    
    return data
    

def prepare_data_for_company(data_dir, start_date, end_date, symbol, finnhub_client, with_basics=True):
    
    data = get_returns(start_date, end_date, symbol)
    data = get_news(symbol, data, finnhub_client)
    
    if with_basics:
        data = get_basics(start_date, symbol, data, finnhub_client)
        data.to_csv(f"{data_dir}/{symbol}_{start_date}_{end_date}.csv")
    else:
        data['Basics'] = [json.dumps({})] * len(data)
        data.to_csv(f"{data_dir}/{symbol}_{start_date}_{end_date}_nobasics.csv")
    
    return data