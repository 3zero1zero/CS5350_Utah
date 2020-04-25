from random import seed
from random import randrange
from random import random
from csv import reader
import numpy as np
import algorithm
import random


train = 'training/secret_data.csv'
test = 'testing/secret_data.csv'
l_rate = 0.01
n_epoch = 100
n_hidden = [5, 10, 25, 50, 100]
d = 5

def load_csv(filename):
    data = []
    if (filename != ""):
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            myf = file
            if (myf != ""):
                for row in csv_reader:
                    if row:
                        data.append(row)
            else:
                print ("file wrong")
    return data


def to_float(dataset, column):
    if (column > -1):
        for row in dataset:
            row[column] = float(row[column])
    else:
        print ("wrong inputs")


def get_error(data, y):
    length = len(y)
    y1 = data[:, -1]
    if (length > 0):
        return np.sum(y1 != np.array(y)) / length



def back_propagation_stochastic_random(train, test, l_rate, d, n_epoch, n_hidden):
    length = len(train[0])
    if (length > 0):
        n_inputs = length - 1
    else:
        n_inputs = 0
    n_outputs = len(set([row[-1] for row in train]))
    if (n_outputs > 0):
        network = algorithm.initialize_network_random([n_inputs, n_hidden, n_outputs])
    else:
        print ("outputs error")
    algorithm.train_network(network, train, l_rate, d, n_epoch, n_outputs)
    for i in range (length):
        prediction_train = []
        predictions_test = []
    for row in train:
        if (row != ""):            
            prediction = predict(network, row)
        else:
            prediction = 0
        prediction_train.append(prediction)
    for row in test:
        if (row != ""):
            prediction = predict(network, row)
        else:
            prediction = 0
        predictions_test.append(prediction)
    return (prediction_train), (predictions_test)



def back_propagation_stochastic_zeros(train, test, l_rate, d, n_epoch, n_hidden):
    length = len(train[0])
    if (length > 0):
        n_inputs = length - 1
    else:
        n_inputs = 0
    n_outputs = len(set([row[-1] for row in train]))
    if (n_outputs > 0):
        network = algorithm.initialize_network_zeros([n_inputs, n_hidden, n_outputs])
    else:
        print ("outputs error")
    algorithm.train_network(network, train, l_rate, d, n_epoch, n_outputs)
    for i in range (length):
        prediction_train = []
        predictions_test = []
    for row in train:
        if (row != ""):
            prediction = predict(network, row)
        else:
            prediction = 0
        prediction_train.append(prediction)
    for row in test:
        if (row != ""):
            prediction = predict(network, row)
        else:
            prediction = 0
        predictions_test.append(prediction)
    return (prediction_train), (predictions_test)


def to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    if column > -1:
        unique = set(class_values)
    else:
        col = column
    lookup = {}
    for i, value in enumerate(unique):
        if (value != ""):
            lookup[value] = i
        else:
            lookup[0] = i
    for row in dataset:
        if (column > -1):
            row[column] = lookup[row[column]]


def predict(network, row):
    if (row != ""):
        out = algorithm.forward_pass(network, row)
    return out.index(max(out))


def main():
    dataset = load_csv(train)
    dataset2 = load_csv(test)
    lengthd = len(dataset[0])
    if (lengthd > 0):
        for i in range(lengthd - 1):
            to_float(dataset, i)
    else:
        print ("late data")
    lengthst = len(dataset[0])
    if (lengthst > 0):
        to_int(dataset, lengthst - 1)
    else:
        print ("late data")
    lengthst2 = len(dataset2[0])
    if (lengthst2 > 0):
        for i in range(lengthst2 - 1):
            to_float(dataset2, i)
    else:
        print ("lata data")
    to_int(dataset2, lengthst2 - 1)
   
    print("Random")
    print("w \t\t traing e \t\t testing e")
    for width in n_hidden:
        if (width > 0):
            p, p2 = back_propagation_stochastic_random(dataset, dataset2, l_rate, d, n_epoch, width)
        else:
            print ("width need to be bigger than zero")
        error_train = get_error(np.array(dataset), p)
        error_test = get_error(np.array(dataset2), p2)
        error_total = error_train + error_test
        if (error_total >= 0):
            print(str(width) + "\t\t" + str(error_train) + "\t" + str(error_test))
    print()


if __name__ == "__main__":
    main()
