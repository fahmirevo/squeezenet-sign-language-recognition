import numpy as np


def nchw(data):
    return data.reshape((data.shape[0], 1, *data.shape[1:]))


def train_test_split(X, Y, train_size=0.75):
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)

    splitter = int(len(X) * train_size)

    X_train, X_test = X[idxs[:splitter]], X[idxs[splitter:]]
    Y_train, Y_test = Y[idxs[:splitter]], Y[idxs[splitter:]]

    return X_train, Y_train, X_test, Y_test


def clean_save(X, Y, name=''):
    X = nchw(X)

    np.save('dataset/X_' + name, X)
    np.save('dataset/Y_' + name, Y)


if __name__ == '__main__':
    X = np.load('dataset/X.npy')
    Y = np.load('dataset/Y.npy')

    X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

    clean_save(X_train, Y_train, 'train')
    clean_save(X_test, Y_test, 'test')
