import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
paths = []
range_of_a = [str(i).zfill(2) for i in np.arange(1, 20, 1)]
range_of_p = [str(i).zfill(1) for i in np.arange(1, 9, 1)]
range_of_s = [str(i).zfill(2) for i in np.arange(1, 61, 1)]

labels = []
for a in range_of_a:
    for p in range_of_p:
        for s in range_of_s:
            labels.append(int(a))
            paths.append(f"dataset/a{a}/p{p}/s{s}.txt")


def generate_data_matrix(method="mean"):
    if method == "mean":
        return method_1()
    elif method == "flatten":
        return method_2()


def method_1():
    no_rows = len(paths)
    X_train = np.zeros((19 * 8 * 48, 45))
    X_test = np.zeros((19 * 8 * 12, 45))
    y_train = np.zeros((19 * 8 * 48, 1))
    y_test = np.zeros((19 * 8 * 12, 1))
    test_index = 0
    train_index = 0
    for i in range(no_rows):
        data = pd.read_csv(paths[i], header=None)
        if (i % 60) < 48:
            X_train[train_index] = np.mean(data.values, axis=0)
            y_train[train_index] = labels[i]
            train_index += 1
        else:
            X_test[test_index] = np.mean(data.values, axis=0)
            y_test[test_index] = labels[i]
            test_index += 1
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test


def method_2():
    no_rows = len(paths)
    X_train = np.zeros((19 * 8 * 48, 125 * 45))
    X_test = np.zeros((19 * 8 * 12, 125 * 45))
    y_train = np.zeros((19 * 8 * 48, 1))
    y_test = np.zeros((19 * 8 * 12, 1))
    test_index = 0
    train_index = 0
    for i in range(no_rows):
        data = pd.read_csv(paths[i], header=None)
        if (i % 60) < 48:
            X_train[train_index] = data.values.flatten()
            y_train[train_index] = labels[i]
            train_index += 1
        else:
            X_test[test_index] = data.values.flatten()
            y_test[test_index] = labels[i]
            test_index += 1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, y_train, X_test, y_test
