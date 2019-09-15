from usedImports import pd
import datetime as dt
import os
import urllib.request
import json


def load_data(data_source):
    if data_source == "alphavantage":
        api_key = "W9PE4VEM0NZUFON1"
        ticker = "AAL"  # American Airline Data

        # Stock Markel AAL Data for the last 20 years
        url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s" % (
            ticker, api_key)

        file_to_save = "Data/stock_market_data_%s.csv" % ticker  # filename

        # Save the file
        if not os.path.exists(file_to_save):
            with urllib.request.urlopen(url_string) as url:
                data = json.loads(url.read().decode())

                # extract the stock market data
                data = data["Time Series (Daily)"]
                df = pd.DataFrame(
                    columns=["Date", "Low", "High", "Close", "Open"])

                for k, v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    data_row = [date.date, float(v["3. low"]), float(
                        v["2. high"]), float(v["4. close"]), float(v["1. open"])]
                    df.loc[-1, :] = data_row
                    df.index = df.index + 1

                print('Data is saved to :%s' % file_to_save)
                df.to_csv(file_to_save)

       # if the data exists load the data
        else:
            print("Loading file from %s" % file_to_save)
            df = pd.read_csv(file_to_save)

    # read the data from Kaggle
    else:
        df = pd.read_csv(os.path.join("Data", "kaggle_data", "Stocks", "hps.us.txt"),
                         delimiter=",", usecols=["Date", "Open", "High", "Low", "Close"])
        print("Loaded Kaggle Dataset")

    # Sort the DataFrame by its Date
    df = df.sort_values("Date")
    return df
