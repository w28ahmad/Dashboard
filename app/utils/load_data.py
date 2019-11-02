import os
from dotenv import load_dotenv
import pandas_datareader.data as web
import sys

def stock_data(start_date, end_date, name, BASEDIR):
    # Connect the path with your '.env' file name
    load_dotenv(os.path.join(BASEDIR, '.env'))
    df = web.DataReader(name=name, data_source='quandl', start=start_date, end=end_date, api_key=os.getenv("quandle_key"))
    return df

'''
if __name__ == "__main__":
    start_date = '2017-04-04'
    end_date = '2018-04-04'
    BASEDIR = os.path.join(os.getcwd(), '..')
    print(stock_data(start_date, end_date, 'AAPL', BASEDIR))
'''
