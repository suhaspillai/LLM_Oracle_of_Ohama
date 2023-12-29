import os
import re
import csv
import math
import time
import json
import random
import finnhub
import datasets
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import defaultdict
from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm
from get_financial_data import prepare_data_for_company
from get_prompt_data import query_gpt4, create_dataset


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--start_date', default='2023-10-01', help='start date for downloading news articles e.g yyyy-MM-DD')
  parser.add_argument('--end_date', default='2023-12-26', help='end date for downloading news articles e.g yyyy-MM-DD')
  parser.add_argument('--env_file', default='.env', help='file that store keys for openai, huggingface, finnhub_key etc')
  parser.add_argument('--list_ticker_symbol', default="MSFT", help='Different stock ticker symbols')
  parser.add_argument('--ds_fname', default='hf_datset', help='name of the dataset to save')
  args = parser.parse_args()
  SYSTEM_PROMPT = "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. " \
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n"

  B_INST, E_INST = "[INST]", "[/INST]"
  B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
  
  START_DATE = args.start_date #"2023-08-31"
  END_DATE = args.end_date #"2023-12-08"
  api_keys={}
  with open(args.env_file) as f_r:
  	for line in f_r:
  		val = line.strip().split("=")
  		
  		api_keys[val[0]] = val[1]
  		

  DATA_DIR = f"./{START_DATE}_{END_DATE}"
  os.makedirs(DATA_DIR, exist_ok=True)
  finnhub_client = finnhub.Client(api_key=api_keys["FINNHUB_KEY"])
  client = OpenAI(api_key = api_keys["OPENAI_KEY"])
  #ticker_sym = ["MSFT", "AAPL", "ASML", "NVDA", "TSLA", "TSM","GOOGL", "AMD", "META", "AMZN"]
  ticker_sym = [sym.strip() for sym in args.list_ticker_symbol.split(',')]
  for symbol in tqdm(ticker_sym):
  	print("=========== Fetching data for {} ===========".format(symbol))
  	prepare_data_for_company(DATA_DIR, START_DATE, END_DATE, symbol, finnhub_client)

  print("=========== Query GPT-4 and get analysis based on the news articles ===========")
  query_gpt4(DATA_DIR, START_DATE, END_DATE, ticker_sym, finnhub_client, client,
             SYSTEM_PROMPT, min_past_weeks=1, max_past_weeks=3, with_basics=True)


  print("=========== Reformat data for llama-2 finetuning ===========")
  dataset_hf = create_dataset(DATA_DIR, START_DATE, END_DATE, ticker_sym, SYSTEM_PROMPT,
  	             B_INST, E_INST, B_SYS, E_SYS, train_ratio=0.8,
  	             with_basics=True)
  dataset_hf.save_to_disk(args.ds_fname)

if __name__ == '__main__':
  main()




  


  

