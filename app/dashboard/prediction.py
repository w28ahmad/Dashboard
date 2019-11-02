import numpy as np
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