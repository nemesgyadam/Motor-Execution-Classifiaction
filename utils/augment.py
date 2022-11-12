import numpy as np


def apply_augment(X, y):
    noise_aug = noise_augment(X)
    shift_aug = shift_augment(X)

    X_aug = np.concatenate((X, noise_aug, shift_aug))
    y_aug = np.concatenate((y, y, y))

    return X_aug, y_aug


def noise_augment(X, ratio=0.01, ratio2=9):

    noise = np.random.normal(0, ratio, X.size)
    comp = np.random.randint(ratio2, size=X.size)
    comp -= ratio2 - 2
    comp = comp.clip(min=0)

    noise = noise * comp
    noise = np.reshape(noise, X.shape)

    augmented = X + noise

    return augmented


def shift_augment(X, ratio=0.5, ratio2=9):
    shift = np.random.normal(0, ratio, size=(X.shape[0], X.shape[1]))

    comp = np.random.randint(ratio2, size=(X.shape[0], X.shape[1]))
    comp -= ratio2 - 2
    comp = comp.clip(min=0)

    shift = shift * comp
    shift = np.repeat(shift[:, :, np.newaxis], X.shape[2], axis=2)

    augmented = X + shift

    return augmented
