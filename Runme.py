from Preprocessing import *
from FiveDayModels import train_lstm_models, train_final_model, five_day_prediction
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


option = input("5 day prediction(5), DRQN trader(DRQN)- ")

if option == "5":

    symbol = input("Symbol of company- ")
    weeks = int(input("How many weeks of data should be taken- "))

    raw_data = query_url(symbol.upper())

    arrays = process_data(raw_data)

    print("Organizing samples\n\n")

    one_x, one_y = split_data(arrays, weeks, 5)
    two_x, two_y = split_data(arrays, weeks, 10)
    four_x, four_y = split_data(arrays, weeks, 20)
    eight_x, eight_y = split_data(arrays, weeks, 40)
    twelve_x, twelve_y = split_data(arrays, weeks, 60)
    twentyfour_x, twentyfour_y = split_data(arrays, weeks, 120)
    fourtyeight_x, fourtyeight_y = split_data(arrays, weeks, 240)

    print("Training models\n\n")

    lstm_models = train_lstm_models(one_x, one_y, two_x, two_y, four_x, four_y, eight_x, eight_y,
                                    twelve_x, twelve_y, twentyfour_x, twentyfour_y, fourtyeight_x, fourtyeight_y)

    final_model = train_final_model(lstm_models, one_x, two_x, four_x, eight_x, twelve_x, twentyfour_x,
                                    fourtyeight_x, fourtyeight_y)

    five_day_prediction(lstm_models, final_model, one_x, two_x, four_x, eight_x, twelve_x, twentyfour_x, fourtyeight_x)

elif option == "DRQN":

    option = input("Train or run? ")

    if option == "train":
        print("TODO")
    elif option == "run":
        print("TODO")