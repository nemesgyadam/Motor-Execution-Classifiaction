import numpy as np
import keras
import os
import random
from utils.signal_utils import DCFilter, Notch, Bandpass, Resample, Normalize
from utils.augment import apply_augment


class SessionGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(self, config, split="train"):
        "Initialization"
        random.seed(42)
        np.random.seed(42)
        self.start_triggers = [1, 2, 3]
        self.end_triggers = [10]
        self.do_augment = split == "train"
        self.data_path = config.data_path
        self.config = config
        self.batch_size = config.batch_size
        self.split = split
        self.load_sessions()
        self.preprocess()
        self.cut_trials()
        # self.shuffle()

        print("shapes", self.X.shape, self.y.shape)
        if self.do_augment:
            self.X, self.y = apply_augment(self.X, self.y)
            print("AUGMENTED", self.X.shape, self.y.shape)
        #self.channels = self.X.shape[-2]

    def load_sessions(self):
        session_paths = []
        self.sessions = []

        for subject in os.listdir(self.data_path):
            for session in os.listdir(os.path.join(self.data_path, subject)):
                if (
                    self.split == "train"
                    and (subject, session) in self.config.train_sessions
                ):
                    session_paths.append(os.path.join(self.data_path, subject, session))
                elif (
                    self.split == "val"
                    and (subject, session) in self.config.val_sessions
                ):
                    session_paths.append(os.path.join(self.data_path, subject, session))
        print(f"Loading data from{session_paths}")
        for session in session_paths:
            self.sessions.append(np.load(os.path.join(session, "data.npy")))

    def shuffle(self):
        "Shuffle the data"
        if self.split == "train":
            c = list(zip(self.X, self.y))
            random.shuffle(c)
            X, y = zip(*c)
            self.X = np.array(X)
            self.y = np.array(y)

    def prep(self, full_session):
        session = full_session[2:, :]
        # print("--")
        # print(session.shape)
        if self.config.DC_filter:
            session = DCFilter(session)

        if self.config.notch:
            session = Notch(session, freq=self.config.notch_freq)

        if self.config.bandpass:
            session = Bandpass(
                session,
                lowcut=self.config.bandpass_freq[0],
                highcut=self.config.bandpass_freq[1],
                order=self.config.order,
            )

        if self.config.resample_to:
            session = Resample(session, self.config.resample_to)

        if self.config.normalize:
            session = Normalize(session)
        return np.concatenate((full_session[:2, :], session), axis=0)

    def preprocess(self):
        "Preprocess the data"
        self.sessions = map(self.prep, self.sessions)

    def cut_trials(self):
        """
        Split the session to trials based on triggers.
        # trigger[0] = timestamp
        # trigger[1] = trigger value
        # trigger[2] = relative time
        """
        X = []
        y = []
        for session in self.sessions:
            triggers = []
            for i in range(session.shape[-1]):
                time_stamp = session[0, i]
                trigger = session[1, i]
                rel_pos = i
                if trigger != 0:
                    triggers.append((time_stamp, trigger, i))
            #print(triggers)

            periods = []
            for i in range(len(triggers) - 1):
                current = triggers[i]
                next = triggers[i + 1]
                if int(current[1]) in self.start_triggers:
                    if int(next[1]) in self.end_triggers:
                        periods.append((current[2], next[2], current[1]))
                    else:
                        print("ERROR: No end trigger found")


            for period in periods:
                # Get the EEG channels only
                # (so remove the first 2 channels, which are time and trigger)
                # Create X, y tuples
                n_channels = session.shape[0] - 2
                sample_rate = 250
                l = (
                    sample_rate * self.config.sample_length
                    if self.config.resample_to is None
                    else self.config.resample_to * self.config.sample_length
                )

                # Corrigate trial length
                # to fix length
                _X = np.zeros((n_channels, sample_rate * self.config.sample_length))
               
                length = min(
                    period[1] - period[0], sample_rate * self.config.sample_length
                )
                _X[:, :length] = session[2:, period[0] : (period[0] + (length))]
                _y = period[2]-1

                X.append(_X)
                y.append(_y)
        self.X = np.array(X)
        self.y = np.array(y)

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
