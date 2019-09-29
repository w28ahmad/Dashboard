import os
from dotenv import load_dotenv
import pandas_datareader.data as web

def stock_data(start_date, end_date, name, BASEDIR):
    # Connect the path with your '.env' file name
    load_dotenv(os.path.join(BASEDIR, '.env'))
    df = web.DataReader(name=name, data_source='quandl', start=start_date, end=end_date, access_key=os.getenv("quandle_key"))
    return df
