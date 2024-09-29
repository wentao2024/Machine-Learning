import numpy as np
import random
import pandas as pd
import os
import math
from sklearn.preprocessing import StandardScaler



def load_data(file, inputs, label='y', intialize_theta_0=False, type='float', scale=False):
    with open(file, 'r', encoding='ISO-8859-1') as f:
            headers = f.readline().strip().split(',')

    #print("headers", headers)
    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i] in inputs]
    l_cols = [i for i in range(len(headers)) if headers[i] == label]
    inputs = np.loadtxt(file, dtype=type, delimiter=',', skiprows=1, usecols=x_cols, encoding='ISO-8859-1')
    labels = np.loadtxt(file, dtype=type, delimiter=',', skiprows=1, usecols=l_cols, encoding='ISO-8859-1')
  
    if (scale):
        scaler = StandardScaler().fit(inputs)
        inputs = scaler.transform(inputs)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if intialize_theta_0:
        inputs = add_theta_0(inputs)
    return inputs, labels

def add_theta_0(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.
pyto
    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x



def k_cross_validation(file, n, model, load, predict):
    length = file_length(file + ".csv")

    chunkSize = math.floor((length - 1) / n)
    remainder = (length - 1) % n
    
    adjustment = 1
    print("length", length)
    print("chunk", chunkSize)

    sum = 0

    for i in range(n):
        print(i)
        if (remainder):
            remainder -= 1
        else:
            adjustment = 0
        set = range(1, length)
        sample = random.sample(set, chunkSize + adjustment)
        sample.sort()
        tempFile = file + "_" + str(i)
        data = pd.read_csv(file + ".csv", skiprows=sample, encoding='ISO-8859-1', dtype="str")
        data.to_csv(tempFile + "_train.csv", index=False)
        print(1)
        set = complement_list(set, sample, chunkSize + adjustment)
        complement_sample = complement_list(range(1, length), sample, chunkSize + adjustment)
        print(2)
        data_test = pd.read_csv(file + ".csv", skiprows=complement_sample, encoding='ISO-8859-1', dtype="str")
        data_test.to_csv(tempFile + "_test.csv", index=False)
        print(3)
        x, y = load(tempFile + "_train.csv")
        theta = model(x, y)
        print(4)
        x_test, y_test = load(tempFile + "_test.csv")
        y_pred = predict(x_test, theta)
        print(5)
        mse = np.mean((y_pred - y_test)**2)
        print("MSE on test =", mse)
        sum += mse
        #mse1 = np.mean((predict(x, theta) - y)**2)
        #print("MSE on training set =", mse1)

        #print("sample", len(sample))
        #index = complement_list(range(1, length), sample, chunkSize + adjustment)
        set = complement_list(set, sample, chunkSize + adjustment)
        #print("index", len(index))

    for i in range(n):
        os.remove(file + "_" + str(i) + "_train.csv")
        os.remove(file + "_" + str(i) + "_test.csv")

    print("mean is", sum / n)

def complement_list(set, index_old, length):
    index_new = []
    idx1 = 0 #index in set
    idx2 = 0 #index in index_old

    n = len(set)
    for i in range(n):
        if set[i] == index_old[idx2]:
            index_new += set[idx1:i]
            idx2 += 1
            idx1 = i + 1
            
            if idx2 == length:
                index_new += set[idx1:]  
                break
    return index_new

def file_length(file):
    length = 0
    with open(file, 'r', encoding='ISO-8859-1') as f:
        #headers = f.readline().strip().split(',')
        for line in f:
            length += 1

    return length

def print_test(function):
    function("Hello World!")
