import numpy as np
import scipy.signal as sg
import abc


class ModifierFramework(metaclass=abc.ABCMeta):

    default_probability = 0.1

    def __init__(self, probability=None):
        if probability is None:
            self.probability = self.default_probability
        else:
            self.probability = probability

    def do(self, data):
        mask = self.select(data)
        data[mask] = self.modify(data[mask])

        return data

    def select(self, data):
        n_data = data.shape[0]
        return np.random.random(n_data) < self.probability

    @abc.abstractmethod
    def modify(self, data):
        pass


class Blocker(ModifierFramework):

    default_probability = 0.2

    def modify(self, data):
        n_data = data.shape[0]
        blocker_type = np.random.random(n_data)
        blocker_range = np.random.randint(0, 20, 4)

        mask = blocker_type < 0.3
        data[mask, :, blocker_range[0]:] = 0

        mask = (blocker_type >= 0.3) & (blocker_type < 0.6)
        data[mask, :, :blocker_range[1]] = 0

        mask = (blocker_type >= 0.6) & (blocker_type < 0.8)
        data[mask, :, :, blocker_range[2]:] = 0

        mask = (blocker_type >= 0.8) & (blocker_type < 1)
        data[mask, :, :, :blocker_range[3]] = 0

        return data


class PixelKiller(ModifierFramework):

    th = 0.2

    def modify(self, data):
        mask = np.random.random(data.shape) < self.th
        data[mask] = np.random.uniform(0, 1, data[mask].shape)
        return data


class Rotator(ModifierFramework):

    def modify(self, data):
        n_data = data.shape[0]
        rotate_type = np.random.randint(0, 4, n_data)

        for i in range(4):
            mask = rotate_type >= i
            data[mask] = np.rot90(data[mask], axes=(2, 3))

        return data


class EdgeDetector(ModifierFramework):

    default_probability = 0.4

    def modify(self, data):
        kernel_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
        kernel_x = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])

        for datum in data:
            dy = sg.convolve2d(datum[0], kernel_y, boundary='wrap', mode='same')
            dx = sg.convolve2d(datum[0], kernel_x, boundary='wrap', mode='same')

            dist = (dy ** 2 + dx ** 2) ** 0.5
            dist[dist > 1] = 1

            datum[0] = dist

        return data


class RandomModifier:

    def __init__(self, generator):
        self.generator = generator
        self.modifiers = [Blocker(), PixelKiller(), Rotator()]

    def __call__(self, *args, **kwargs):
        self.generator = self.generator(*args, **kwargs)
        return self

    def __next__(self):
        X, Y = next(self.generator)
        for modifier in self.modifiers:
            X = modifier.do(X)

        return X, Y


@RandomModifier
def data_generator(batch_size=128):
    X = np.load("dataset/X_train.npy")
    Y = np.load("dataset/Y_train.npy")

    idxs = np.arange(len(X))

    while True:
        np.random.shuffle(idxs)
        X_batch = X[idxs[:batch_size]].copy()
        Y_batch = Y[idxs[:batch_size]].copy()

        yield X_batch, Y_batch


def data_iterator(batch_size=128):
    X = np.load("dataset/X_train.npy")
    Y = np.load("dataset/Y_train.npy")

    idxs = np.arange(len(X))

    while True:
        np.random.shuffle(idxs)

        chunks = np.array_split(idxs, (len(idxs) // batch_size) + 1)

        for chunk in chunks:
            yield X[chunk], Y[chunk]
