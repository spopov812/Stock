from keras import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector, Dropout, Flatten
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys


def build_lstm_model(time_steps):

    features = 5

    model = Sequential()

    model.add(LSTM(128, input_shape=(features, time_steps), return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu"))

    model.add(RepeatVector(5))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(TimeDistributed(Dense(1)))
    model.add(Activation('linear'))

    model.compile(

        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_squared_error"]
    )

    return model


def build_final_model():

    model = Sequential()

    model.add(LSTM(128, input_shape=(7, 5), return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(128, activation="relu"))

    model.add(RepeatVector(5))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(TimeDistributed(Dense(1)))
    model.add(Activation('linear'))

    model.compile(

        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_squared_error"]
    )

    return model


# callback to save model
def build_callbacks(name):
    callbacks = []

    # name = "C:\\Users\\Sasha\\PycharmProjects\\StockPrediction\\" + name + "\\{epoch:02d}-{mean_squared_error:.2f}.h5"

    # callbacks.append(ModelCheckpoint(name,
                                     # monitor='mean_squared_error', verbose=0,
                                     # save_best_only=True, save_weights_only=False, mode='auto', period=1))

    return callbacks


def train_lstm_models(one_x, one_y, two_x, two_y, four_x, four_y, eight_x, eight_y,
                                twelve_x, twelve_y, twentyfour_x, twentyfour_y, fourtyeight_x, fourtyeight_y):

    # creating models
    one_model = build_lstm_model(5)
    two_model = build_lstm_model(10)
    four_model = build_lstm_model(20)
    eight_model = build_lstm_model(40)
    twelve_model = build_lstm_model(60)
    twentyfour_model = build_lstm_model(120)
    fourtyeight_model = build_lstm_model(240)

    # print("Model architecture-\n\n")
    # one_model.summary()

    print("\n\n")

    # training models

    sys.stdout.write("\rTraining model 1/7")
    sys.stdout.flush()

    history = one_model.fit(one_x, one_y, epochs=15, batch_size=8, callbacks=build_callbacks("One"), verbose=0)

    print("\n\nFinal loss value- ", history.history['mean_squared_error'][-1])

    sys.stdout.write("\rTraining model 2/7")
    sys.stdout.flush()

    history = two_model.fit(two_x, two_y, epochs=15, batch_size=8, callbacks=build_callbacks("Two"), verbose=0)

    print("\n\nFinal loss value- ", history.history['mean_squared_error'][-1])

    sys.stdout.write("\rTraining model 3/7")
    sys.stdout.flush()

    history = four_model.fit(four_x, four_y, epochs=15, batch_size=8, callbacks=build_callbacks("Four"), verbose=0)

    print("\n\nFinal loss value- ", history.history['mean_squared_error'][-1])

    sys.stdout.write("\rTraining model 4/7")
    sys.stdout.flush()

    history = eight_model.fit(eight_x, eight_y, epochs=15, batch_size=8, callbacks=build_callbacks("Eight"), verbose=0)

    print("\n\nFinal loss value- ", history.history['mean_squared_error'][-1])

    sys.stdout.write("\rTraining model 5/7")
    sys.stdout.flush()

    history = twelve_model.fit(twelve_x, twelve_y, epochs=15, batch_size=8, callbacks=build_callbacks("Twelve"), verbose=0)

    print("\n\nFinal loss value- ", history.history['mean_squared_error'][-1])

    sys.stdout.write("\rTraining model 6/7")
    sys.stdout.flush()

    history = twentyfour_model.fit(twentyfour_x, twentyfour_y, epochs=20, batch_size=8, callbacks=build_callbacks("TwentyFour"), verbose=0)

    print("\n\nFinal loss value- ", history.history['mean_squared_error'][-1])

    sys.stdout.write("\rTraining model 7/7")
    sys.stdout.flush()

    history = fourtyeight_model.fit(fourtyeight_x, fourtyeight_y, epochs=20, batch_size=8, callbacks=build_callbacks("FourtyEight"), verbose=0)

    print("\n\nFinal loss value- ", history.history['mean_squared_error'][-1])

    lstm_models = [one_model, two_model, four_model, eight_model, twelve_model, twentyfour_model, fourtyeight_model]

    return lstm_models


def train_final_model(lstm_models, one_x, two_x,four_x, eight_x, twelve_x, twentyfour_x, fourtyeight_x, fourtyeight_y):

    one_model = lstm_models[0]
    two_model = lstm_models[1]
    four_model = lstm_models[2]
    eight_model = lstm_models[3]
    twelve_model = lstm_models[4]
    twentyfour_model = lstm_models[5]
    fourtyeight_model = lstm_models[6]

    num_samples = len(fourtyeight_x)

    print("\n\nGenerating predictions for final model training...\n\n")

    one_predict = one_model.predict(one_x[:num_samples]).reshape(num_samples, 1, 5)
    two_predict = two_model.predict(two_x[:num_samples]).reshape(num_samples, 1, 5)
    four_predict = four_model.predict(four_x[:num_samples]).reshape(num_samples, 1, 5)
    eight_predict = eight_model.predict(eight_x[:num_samples]).reshape(num_samples, 1, 5)
    twelve_predict = twelve_model.predict(twelve_x[:num_samples]).reshape(num_samples, 1, 5)
    twentyfour_predict = twentyfour_model.predict(twentyfour_x[:num_samples]).reshape(num_samples, 1, 5)
    fourtyeight_predict = fourtyeight_model.predict(fourtyeight_x[:num_samples]).reshape(num_samples, 1, 5)

    x_data = np.concatenate((one_predict, two_predict, four_predict, eight_predict, twelve_predict, twentyfour_predict,
                             fourtyeight_predict), axis=1)

    y_data = fourtyeight_y[:num_samples]

    final_model = build_final_model()

    # print("Model architecture-\n\n")
    # final_model.summary()

    print("\n\nTraining final model...\n\n")

    history = final_model.fit(x_data, y_data, epochs=20, batch_size=1, callbacks=build_callbacks("Final"), verbose=0)

    print("\n\nFinal loss value- ", history.history['mean_squared_error'][-1])

    return final_model


def five_day_prediction(lstm_models, final_model, one_x, two_x, four_x, eight_x, twelve_x, twentyfour_x, fourtyeight_x):

    one_model = lstm_models[0]
    two_model = lstm_models[1]
    four_model = lstm_models[2]
    eight_model = lstm_models[3]
    twelve_model = lstm_models[4]
    twentyfour_model = lstm_models[5]
    fourtyeight_model = lstm_models[6]

    num_samples = 1

    one_predict = one_model.predict(one_x[:num_samples]).reshape(num_samples, 1, 5)
    two_predict = two_model.predict(two_x[:num_samples]).reshape(num_samples, 1, 5)
    four_predict = four_model.predict(four_x[:num_samples]).reshape(num_samples, 1, 5)
    eight_predict = eight_model.predict(eight_x[:num_samples]).reshape(num_samples, 1, 5)
    twelve_predict = twelve_model.predict(twelve_x[:num_samples]).reshape(num_samples, 1, 5)
    twentyfour_predict = twentyfour_model.predict(twentyfour_x[:num_samples]).reshape(num_samples, 1, 5)
    fourtyeight_predict = fourtyeight_model.predict(fourtyeight_x[:num_samples]).reshape(num_samples, 1, 5)

    x_data = np.concatenate((one_predict, two_predict, four_predict, eight_predict, twelve_predict, twentyfour_predict,
                             fourtyeight_predict), axis=1)

    prediction = final_model.predict(x_data)

    print("\n\nPrediction for next five days-\n")

    for i in range(5):

        print("Day %d- %.2f" % ((i + 1), prediction[0][i]))