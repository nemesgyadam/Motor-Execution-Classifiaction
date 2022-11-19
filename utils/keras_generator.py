import numpy as np
import keras
import os
import random
from utils.signal_utils import DCFilter, Notch, Bandpass, Resample, Normalize
from utils.augment import apply_augment


class Generator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(self, config, split="train"):
        "Initialization"
        random.seed(42)
        np.random.seed(42)
        self.do_augment = split == "train"
        self.data_path = config.data_path
        self.config = config
        self.batch_size = config.batch_size
        self.split = split
        self.load_sessions()
        self.shuffle()
        self.preprocess()

        print(self.X.shape, self.y.shape)
        if self.do_augment:
            self.X, self.y = apply_augment(self.X, self.y)
            print("AUGMENTED", self.X.shape, self.y.shape)
        self.channels = self.X.shape[-2]

    def load_sessions(self):
        sessions = []
        data = {}
        for c in self.config.classes:
            data[c] = []

        for subject in os.listdir(self.data_path):
            for session in os.listdir(os.path.join(self.data_path, subject)):
                if (
                    self.split == "val"
                    and (subject, session) in self.config.val_sessions
                ):
                    sessions.append(os.path.join(self.data_path, subject, session))
                elif (
                    self.split == "train"
                    and (subject, session) not in self.config.val_sessions
                ):
                    sessions.append(os.path.join(self.data_path, subject, session))

        for session in sessions:
            for c in self.config.classes:
                data[c].append(np.load(os.path.join(session, c + ".npy")))

        for c in self.config.classes:
            data[c] = np.concatenate(data[c], axis=0)

        X = np.concatenate(list(data.values()), axis=0)
        labels = []
        for i in range(len(self.config.classes)):
            labels.append(np.full((data[self.config.classes[i]].shape[0]), i))
        y = np.concatenate(labels, axis=0)
        print(X.shape, y.shape)

        self.X = X
        self.y = y

    def shuffle(self):
        "Shuffle the data"
        if self.split == "train":
            c = list(zip(self.X, self.y))
            random.shuffle(c)
            X, y = zip(*c)
            self.X = np.array(X)
            self.y = np.array(y)

    def preprocess(self):
        "Preprocess the data"
        if self.config.DC_filter:
            self.X = DCFilter(self.X)

        if self.config.notch:
            self.X = Notch(self.X, freq=self.config.notch_freq)

        if self.config.bandpass:
            self.X = Bandpass(
                self.X,
                lowcut=self.config.bandpass_freq[0],
                highcut=self.config.bandpass_freq[1],
                order=self.config.order,
            )

        # CUT START AND END
        self.X = self.X[:, :, 50:450]

        if self.config.resample_to:
            self.X = Resample(self.X, self.config.resample_to)

        if self.config.normalize:
            self.X = Normalize(self.X)

    def expand_X(self):
        self.X = np.expand_dims(self.X, axis=-1)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        X = self.X[index * self.batch_size : (index + 1) * self.batch_size]
        y = self.y[index * self.batch_size : (index + 1) * self.batch_size]
        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.shuffle()
