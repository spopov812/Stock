import pandas as pd
import numpy as np
from urllib.request import urlopen
import datetime


# building api query
def query_url(symbol):

    print("\n\nUrl query.\n\n")

    key = "V37AF2MUTPQTINXB"

    query = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full" \
            "&apikey=%s&datatype=csv" % (symbol, key)

    raw_data = pd.read_csv(urlopen(query))

    return raw_data


# cleaning data to include full week of data
def process_data(raw_data):

    mk_open, mk_close, mk_high, mk_low, mk_volume = raw_data["open"], raw_data["close"], raw_data["high"], \
                                                    raw_data["low"], raw_data["volume"]

    arrays = [np.array(mk_open.tolist()), np.array(mk_close.tolist()), np.array(mk_high.tolist()),
             np.array(mk_low.tolist()), np.array(mk_volume.tolist())]

    return arrays


'''
# getting day of the week for an entry
def get_day_of_week(entry):

    timestamp = entry["timestamp"]

    ts_as_list = timestamp.tolist()
    ts_as_list = ts_as_list[0].split(" ")

    date, hour = ts_as_list[0].split("-"), ts_as_list[1].split(":")[0]

    year, month, day = date[0], date[1], date[2]

    day_as_num = datetime.date(year, month, day).weekday()

    return day_as_num, hour
'''


def split_data(arrays, weeks, days):
    data_points = weeks * 5

    mk_open, mk_close, mk_high, mk_low, mk_volume = arrays[0], arrays[1], arrays[2], arrays[3], arrays[4]

    # print("Before slicing-\n\n", mk_open[:5])

    y_list = []
    x_list = []

    y_temp = mk_high

    # slicing data
    mk_open, mk_close, mk_high, mk_low, mk_volume = mk_open[5:data_points], mk_close[5:data_points], \
                                                    mk_high[5:data_points], mk_low[5:data_points], \
                                                    mk_volume[5:data_points]

    # print("After slicing-\n\n", mk_open[:5])

    num_samples = len(mk_open) - (days - 1)

    for i in range(num_samples):
        one_sample = []

        temp = mk_open[i:i + days]
        one_sample.append(temp[::-1])

        temp = mk_close[i:i + days]
        one_sample.append(temp[::-1])

        temp = mk_high[i:i + days]
        one_sample.append(temp[::-1])

        temp = mk_low[i:i + days]
        one_sample.append(temp[::-1])

        temp = mk_volume[i:i + days]
        one_sample.append(temp[::-1])

        x_list.append(one_sample)

        temp = y_temp[i:5 + i]
        temp = temp[::-1]
        y_list.append(temp)
        '''
        print("What was appended to x_data-\n\n", one_sample)
        print("\n\nWhat was appended to y_data-\n\n", temp)
        print("\n\nCurrent X Data-\n\n", x_list)
        wait = input("Continue")
        '''
    x, y = np.array(x_list), np.array(y_list).reshape(len(y_list), 5, 1)

    # print(x)

    return x, y
