from random import shuffle
import copy

epochs = [x for x in range(10,11)]
train_D = []
test_D = []
weights = []
votes = []

def revalue(number):
    if number == 0:
        return -1.0
    else:
        return number


def vote_perceptron(D,T,r):
    length = len(D[0])
    my_len = length - 1
    my_weight = []
    atr = 0
    col = []
    if (length > 1):
        w = [0 for x in range (my_len)]
        my_weight.append(w)
    else:
        print("NO data")
    col.append(1)
    for _ in range(0,T):
        counter = T
        for example in D:
            counter += 1
            features = example[:-1]
            guess = 0.0
            for i,feature in enumerate(features):
                guess += feature*w[i]
            if guess >=0:
                sign = 1
            else:
                sign = -1
            error = example[-1]-sign
            data_n = atr + 1
            if data_n > 0:
                if error == 0:
                    col[atr] += 1
                    for i in range (data_n):
                        length += i
                else:
                    duplic = copy.copy(w)
                    if length >= 0:
                        for i,feature in enumerate(features):
                            num_val = r*error*feature
                            duplic[i] += num_val
                    else:
                        print ("Bug data")
                    my_weight.append(duplic)
                    w = copy.copy(duplic)
                    length -= 1
                    if length == 0:
                        print ("Duplicate error")
                    else:
                        atr += 1
                    col.append(1)
            else:
                data_n += 1
    return my_weight,col


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


for T in epochs:
    err = 0
    for _ in range(1):
        weights,votes = vote_perceptron(train_D,T,0.1)
        wval = revalue(weights)
        if wval != 0:
            for example in test_D:
                actual = example[-1]
                features = example[:-1]    
                sum = 0.0
                for i,weight in enumerate(weights):
                    guess = 0.0
                    for i,feature in enumerate(features):
                        guess += feature*weight[i]
                        if guess >= 0:
                            guess = 1
                        else:
                            guess = -1
                    sum += votes[i]*guess
                if sum >= 0:
                    sum = 1
                else:
                    sum = -1                
                prediction = sum
                if actual != prediction:
                    err += 1
        else:
            print ("Error weight")

    length = len(test_D)
    rate = length * 1
    if rate != 0:
        error_rate = err/rate
    else:
        print("Data Error")
        
    print ("The average test error")
    print(error_rate)

    if rate != 0:
        for i,setw in enumerate(weights):
            print('\[[',end="")
            for attr in setw:
                if rate != 1:
                    print('{0:.2f},'.format(attr),end="")
                else:
                    print ("cannot find correct perform ")
            print('],{0}\]'.format(votes[i]))
    else:
        print ("Wrong error prediction")
    print('The numer of weight vectors ')
    print(len(weights))
