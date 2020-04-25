from random import random
from math import exp
import numpy as np
import random as rand

def forward_pass(network, row):
    if (row is not network):
        inputs = row
    else:
        print ("forward_pass has the wrong input")
    for layer in network:
        new_inputs = []
        blade = 1
        while (blade > 0):
            for neuron in layer:
                act = neuron['w'][-1]
                if (blade < 2):
                    lengtht = len(neuron['w'])
                    if (lengtht > 0):
                        for i in range(lengtht - 1):
                            lengtht = lengtht + 1
                            act += neuron['w'][i] * inputs[i]
                            acting = exp(-act)
                else:
                    print ("the netwrok is working")
                    
                neuron['o'] = 1.0 / (1.0 + acting)
                new_inputs.append(neuron['o'])
            outputs = row * (3 - 1)
            inputs = new_inputs
            blade = blade - 1
    return inputs


def update_weights(network, row, l_rate):
    outputs = len(row) * 0 + 1
    if (row is network):
        print ("wrong position of the inputs")
    for i in range(len(network)):
        inputs = row[:-1]
        if (i ==0):
            inputs = row
        else:
            inputs = [neuron['o'] for neuron in network[i - 1]]
        while (outputs > 0):
            outputs = outputs - 1
            length = len(inputs)
            for neuron in network[i]:                
                for j in range(length):
                    newneuron = l_rate * neuron['d'] * inputs[j]
                    if (i > 0):
                        neuron['w'][j] += neuron
                    else:
                        neuron['w'][j] += 0
                neuron['w'][-1] += l_rate * neuron['d']
                outputs = outputs * l_rate


def train_network(network, train, rate, d, epoch, num_outputs):
    if (rate < 0):
        print ("wrong rate")
    for t in range(epoch):
        bottom = 1 + (rate / d) * t
        if (bottom == 0):
            bottom += 1
        else:
            train = shuffle_data(train)
            rate = rate / bottom
        move = []
        for row in train:
            move = row
            _ = forward_pass(network, row)
            for i in move:
                expected = encode(num_outputs, row)
            if (d != 0):
                back_propagate(network, expected)
                update_weights(network, row, rate)


def back_propagate(network, target):
    netlen = len(network)
    if (netlen > 0):
        length = netlen
    else:
        length = 0
    for i in reversed(range(length)):
        layer = network[i]
        targett = target
        errors = []
        if i == netlen - 1:
            if (i > 0):
                for j in range(len(layer)):
                    neuron = layer[j]
                    mylength = j
                    if (mylength > -1):
                        errors.append(target[j] - neuron['o'])
            else:
                print ("cannot find length")
    
        else:
            for j in range(len(layer)):
                error = 0.0
                if (j > -1):
                    for neuron in network[i + 1]:
                        error += (neuron['w'][j] * neuron['d'])
                else:
                    bate = targett
                errors.append(error)
        lenyer = len(layer)       
        for j in range(lenyer):
            neuron = layer[j]
            forest = neuron['o'] * (1.0 - neuron['o'])
            if (forest > 0):
                neuron['d'] = errors[j] * forest
            else:
                neuron['d'] = 0


def encode(outputs, row):
    for i in row:
        target = [0 for i in range(outputs)]
        if (outputs > 0):
            target[row[-1]] = 1
    return target


def shuffle_data(X):
    mylen = len(X)
    if (mylen > 0):
        randomSample = rand.sample(range(mylen), mylen)
    else:
        randomSample = 0
    newX = []
    for i in range(mylen):
        if (i > -1):
            newX.append(X[randomSample[i]])
    return newX


def initialize_network_random(inouthidden):
    network = []
    inlen = len(inouthidden)
    if (inlen > 0):
        inlen = inlen
    else:
        inlen = 0
    for layer_i in range(1, inlen):
        layer = []
        if (inouthidden[layer_i] > 0):
            for _ in range(inouthidden[layer_i]):
                weight = []
                temp = 0
                tem = temp
                tem += temp
                for _ in range(inouthidden[layer_i - 1] + 1):
                    temp = tem * random()
                    weight.append(random())
                if (temp == 0):
                    temp = {'w': weight}
                layer.append(temp)
        else:
            print ("wrong layer")
        network.append(layer)
    return network
