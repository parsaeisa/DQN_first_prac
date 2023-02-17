from logging import ModifiedTensorBoard
from blob import BlobEnv

import numpy as np
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque

import time
import os
import random

MODEL_NAME = "TEST"
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 10_000
MINIBATCH_SIZE = 64
DISCOUNT = 0.9
UPDATE_TARGET_EVERY = 5

MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


class DQNAgent:
    def __init__(self):
        # For every single step that this agent takes, we call model.Predict.

        # Main model - gets trained in every step
        self.model = self.create_model()

        # Target model - this predicts in every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.target_update_counter = 0

        # We take a random sample from this memory and give it to the network as a batch
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE))
        model.add(Activation("Relu"))
        model.add(MaxPool2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("Relu"))
        model.add(MaxPool2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(lr=.001), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        # transitions is our observation from , It consists of:
        # action, reward, next_state, done ( whether it was done or not )
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        x = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) \
                in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(x)/255, np.array(y)/255, batch_size=MINIBATCH_SIZE
                       , verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        # * Shuffle is false because we already randomly sampled data from replay memory

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

env = BlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')