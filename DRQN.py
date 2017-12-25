import numpy as np
from keras.layers import Dense
from keras import Sequential
from urllib.request import urlopen
import pandas as pd


class TrainingAgent:

    # constructor
    def __init__(self):

        self.memory = []

        # learning hyperparameters
        self.gamma = 0.85   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 # exploration decay rate
        self.learning_rate = 1e-4
        self.batch_size = 72

        self.price_window = 30

        self.model = self.build_model()

        self.price_history = []

    # creates neural network for approximating q function
    def build_model(self):

        model = Sequential()

        model.add(Dense(100, input_shape=(200,), activation="linear"))

        model.add(Dense(50, activation="linear"))

        model.add(Dense(3, activation="linear"))

        model.compile(

            optimizer="sgd", lr=self.learning_rate,
            loss="mse",
            metrics=["mse"]

        )

        return model

    # appends important data into memory
    def remember(self, state, next_state, reward, action):

        self.memory.append((state, next_state, reward, action))

    # replays and learns from memories
    def replay(self):

        minibatch = self.memory[-self.batch_size:]

        for state, next_state, reward, action in minibatch:

            target_cat = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target = self.model.predict(state)

            target[0][action] = target_cat

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # performs action through neural network or random value
    def action(self, state):

        if np.random.rand() <= self.epsilon:

            return np.random.randint(3)

        actions = self.model.predict(state)

        return np.argmax(actions)

    # returns price at beginning of sequence
    def get_price_window(self):

        return self.price_history[-self.price_window]

    # calculates reward function
    def calculate_reward(self, action):

        t_price, t_minus_one_price = self.price_history[-1], self.price_history[-2]

        two_day_ratio = (t_price - t_minus_one_price) / t_minus_one_price

        left_half = 1 + (np.sign(action) * two_day_ratio)
        right_half = t_minus_one_price / self.get_price_window()

        return left_half * right_half

    # processes historical price data for correct input into neural network
    def process_x_data(self):

        x_data = []

        for i in range(200):

            x_data.append(self.price_history[-i] - self.price_history[- (i + 1)])

        x_data = np.array(x_data)

        return x_data

    # fetches .csv data
    def get_data(self):

        key = "V37AF2MUTPQTINXB"

        query = "https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=" \
                "USD&apikey=%s&datatype=csv" % key

        raw_data = pd.read_csv(urlopen(query))

        raw_data.to_csv("data.csv")

    # function that starts q learning
    def model_init(self):


