#!/bin/bash

TICKER_SYMBOL="MSFT, AAPL, ASML, NVDA, TSLA, TSM,GOOGL, AMD, META, AMZN"
START_DATE='2023-01-01'
END_DATE='2023-12-26'

python3 prepare_data.py --list_ticker_symbol "MSFT, AAPL, ASML, NVDA, TSLA, TSM,GOOGL, AMD, META, AMZN" --start_date $START_DATE --end_date $END_DATE 

