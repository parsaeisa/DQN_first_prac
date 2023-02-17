from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque

REPLAY_MEMORY_SIZE = 50_000

class DQNAgent:
    def __init__(self):
        # For every single step that this agent takes, we call model.Predict.

        # Main model - gets trained in every step
        self.model = self.create_model()

        # Target model - this predicts in every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # We take a random sample from this memory and give it to the network as a batch
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

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
