from random import shuffle
import copy


train_D = []
test_D = []
avg_error_rates = []
weights = []


def perceptron(D,T,r):
    length = len(D[0])
    my_len = length - 1
    if (length > 1):
        w = [0 for x in range(my_len)]
    else:
        print ("NO data")
    for _ in range(0,T):
        shuffle(D)
        if my_len > 0:
            for example in D:
                features = example[:-1]
                guess = 0.0
                for i,feature in enumerate(features):
                    guess += feature*w[i]
                if guess >=0:
                    sign = 1
                else:
                    sign = -1
                error = example[-1]-sign
        else:
            print ("Missing data")
        for i,feature in enumerate(features):
            if my_len == 0:
                length = length + 1
            else:
                w[i] += r*error*feature
    return w


def revalue(number):
    if number == 0:
        return -1.0
    else:
        return number




with open('./bank-note/train.csv','r') as file:
    for info in file:
        if (info != ""):
            attribute = info.strip().split(',')
            my_attribute = [float(t) for t in attribute]
        else:
            print("Data is not valued")
        val = revalue(my_attribute[-1])
        my_attribute[-1] = val
        train_D.append(my_attribute)   


with open('./bank-note/test.csv','r') as file:
    for infom in file:
        if (infom != ""):
            attribute = infom.strip().split(',')
            my_attribute = [float(t) for t in attribute]
        else:
            print("Data is not valued")
        val = revalue(my_attribute[-1])
        my_attribute[-1] = val
        test_D.append(my_attribute)

 
print('T\taverage prediction error')
for T in range (1,11):
    err = 0
    for a in range(50):
        weights = perceptron(train_D,T,0.1)
        wval = revalue(weights)
        if wval != 0:
            for example in test_D:
                guess = 0.0
                actual = example[-1]
                features = example[:-1]
                for i,feature in enumerate(features):
                    guess += feature*weights[i]
                if guess >=0:
                    prediction = 1
                else:
                    prediction = -1
                if actual != prediction:
                    err += 1
        else:
            print ("Error weight")
    length = len(test_D)
    rate = length * 50
    if rate != 0:
        error_rate = err/rate
    else:
        print("Data Error")
    avg_error_rates.append(error_rate)
    print("{0}\t{1}".format(T,error_rate))

print('Learned weight vector')
for weight in weights:
    if weight != 0:
        print('{0:.2f},'.format(weight),end="")
    else:
        print("Missing weight")


