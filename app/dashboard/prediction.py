import numpy as np
import pandas as pd
from pymongo import MongoClient

from dateutil import parser
# BDay is business day, not birthday...
from pandas.tseries.offsets import BDay


'''
The job of the sequencer is to take in the timestamps(x), data(y) and an integer n, and output n y-values in the
following format [[[], [], []], [[], [], []]].
'''
def data_sequencer(data, n):
    data = np.array(data)
    sequenced_data = []
    
    for i in range(n):
        sequenced_data.append(data[-21-i:-i-1].reshape(-1, 1))
        
    return np.array(sequenced_data)

'''
For each of those n y-values also return an array of dates(x) which is one more than the latest date in any of those
n sets
'''
def time_sequencer(time_data, n):
    sequenced_data = []
    
    for i in range(n):
        date = str(time_data[-i-1])
        sequenced_data.append(parser.parse(date)+BDay(1))
        
    return sequenced_data

'''
connects to mongodb
returns x (date), y1 (polarity), y2 (Subjectivity) 
'''
def sentiment_sequencer():
    import os, sys
    from dotenv import load_dotenv

    BASEDIR = os.path.join(os.getcwd(), "./")
    sys.path.append(BASEDIR)

    #Load environment Variables
    load_dotenv(os.path.join(BASEDIR, '.env'))
    
    mongodb_url = os.getenv("MONGODB_URL")

    # Connect and save data to mongodb
    client = MongoClient(mongodb_url)
    db=client['Tweet_Sentiment']
    coll = db["sentiment_data_amazon"]
    
    # Storing data in arrays
    dates = []
    polarity = []
    subjectivity = []
    tweet = []
    
    for x in coll.find():
        dates.append(x['date'])
        polarity.append(x['polarity'])
        subjectivity.append(x['subjectivity'])
        tweet.append((x['tweet']))
        
    # Create a dataFrame
    data = pd.DataFrame(
        data = {'Date':dates, 
                'polarity':polarity, 
                'subjectivity':subjectivity, 
                'tweet':tweet
            }
        )
    
    # Make sure that the values are sorted by date
    data.sort_values(by='Date', inplace=True)
    
    # Remove all 0 values
    index = data[(data['polarity']==0) & (data['subjectivity']==0)].index
    data.drop(data.index[index], inplace=True)
    
    return data