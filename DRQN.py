import numpy as np
from keras.layers import Dense, LSTM, Flatten
from keras import Sequential
from collections import deque
import random


class TrainingAgent:

    def __init__(self):

        self.memory = deque(300)

        # learning hyperparameters
        self.gamma = 0.85   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-4
        self.batch_size = 64

        self.price_window = 30

        self.model = self.build_model()

        self.price_history = []

    def build_model(self):

        model = Sequential()

        model.add(Dense(100, input_shape=(200,), activation="linear"))

        model.add(Dense(50, activation="linear"))

        model.add(Dense(3, activation="linear"))

        model.compile(

            optimizer="adam", lr=self.learning_rate,
            loss="mse",
            metrics=["mse"]

        )

        return model

    def remember(self, state, next_state, reward, action):

        self.memory.append((state, next_state, reward, action))

    def replay(self):

        minibatch = self.memory[-self.batch_size:]

        for state, next_state, reward, action in minibatch:

            target_cat = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target = self.model.predict(state)

            target[0][action] = target_cat

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def action(self, state):

        if np.random.rand() <= self.epsilon:

            return np.random.randint(3)

        actions = self.model.predict(state)

        return np.argmax(actions)

    def get_price_window(self):

        return self.price_history[-self.price_window]

    def calculate_reward(self, action):

        t_price, t_minus_one_price = self.price_history[-1], self.price_history[-2]

        two_day_ratio = (t_price - t_minus_one_price) / t_minus_one_price

        left_half = 1 + (np.sign(action) * two_day_ratio)
        right_half = t_minus_one_price / self.get_price_window()

        return left_half * right_half

    def process_x_data(self):

        x_data = []

        for i in range(200):

            x_data.append(self.price_history[-i] - self.price_history[- (i + 1)])

        x_data = np.array(x_data)

        return x_data

    def train(self):

