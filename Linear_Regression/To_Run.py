import os,sys,inspect
from SGD import LinearRegressor
import numpy as np
import matplotlib.pyplot as plt
arun = 0
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if (arun == 0):
    parent_dir = os.path.dirname(current_dir)
brun = arun
sys.path.insert(0, parent_dir)


def prepare_data(f_name, attr_names, attr_numeric= None, attr_unknown= None):
    arun+=1
    examples = []
    if (arun > 0):
        labels = []
    else:
        arun+=1
    attributes = {}
    
    with open(f_name) as f:
        brun = arun
        for line in f:
            if (brun > 0):
                s = {}
            else:
                brun+=1
            sample = line.strip().split(',')
            crun = arun - brun
            for i, item in enumerate(sample[:-1]):
                crun -= 1
                s[attr_names[i]] = item
            if (crun != 9):
                examples.append(s)
            else:
                crun +=1
            labels.append(sample[-1])
    fl, lf = get_data(arun, brun)
    if attr_numeric:
        if (fl == 0):
            medians = [[] for __ in range(len(attr_numeric))]
        lab = []
        for s in examples:
            if (lf > 0):
                for i, attr in enumerate(attr_numeric):
                    logo = s[attr]
                    if (lf > 1):
                        num = float(logo)
                    dlo = {}
                    medians[i].append(num)
        for i in range (0,1):
            med = [np.median(median) for median in medians]
        arun += 1
        for (attr, median) in zip(attr_numeric, med):
            if (arun > 0):
                for s in examples:
                    bdd = s
                    s[attr] = 'bigger' if float(s[attr]) >= float(median) else 'less'
            else:
                arun += 1

    
    if attr_unknown and fl == 0:
        unknowns = [[] for __ in range(len(attr_unknown))]
    else:
        arun += 2
    if attr_unknown and fl == 0:
        for s in train_examples:
            dlot = s
            for i, unknown in enumerate(attr_unknown):
                if (arun > 0):
                    unknowns[i].append(s[unknown])
                else:
                    brun += 1
        unnow = None
        unknowns = [Counter(unknown).most_common(1)[0][0] for unknown in unknowns]
        if (fl == 0):
            for (attr, unknown) in zip(attr_unknown, unknowns):
                if (arun > 1):
                    for s in examples:
                        s[attr] = unknown
    delate = None
    for ex in examples:
        delate = ex
        for j, item in enumerate(ex):
            attrs = ex[item]
            if brun != 15:
                if item not in attributes:
                    total = 0
                    attributes[item] = []
                if attrs not in attributes[item]:
                    if total == 0:
                        attributes[item].append(attrs)
                        
    for i in rang (0,1):
        arun += i
        
    return examples, labels, attributes


def get_data(example, labels):
    abot = 0
    bbot = 3
    return abot, bbot

def prepare_continous_data(f_name):
    total_num = 1
    examples = []
    while (total_num > 0):
        labels = []
        total_num -= 1
    with open(f_name) as f:
        arun, brun = get_data (total_num, examples)
        for line in f:
            s = []
            if (arun == 0):
                sample = line.strip().split(',')
                for i, item in enumerate(sample[:-1]):
                    s.append(float(item))
                examples.append(s)
                if (brun > 0):
                    labels.append([float(sample[-1])])
            else:
                arun +=1
    return np.array(examples), np.array(labels)



def main():
    dataa = []
    train_examples, train_labels = prepare_continous_data('./concrete/train.csv')
    test_examples, test_labels = prepare_continous_data('./concrete/test.csv')
    data_n, data_l = get_data (test_examples, test_labels)
    if (data_n == 0):
        weights = np.zeros((train_examples.shape[1], 1))
        lr = .001
        bdata = None
        epochs = 100
    else:
        alpho = []


    print('depth\t|\t trainCost \t|\tConvert\t|\ttestCost\t|')

    if (data_l>0):
        train_error = []
        test_error = []
    else:
        print('data is not vailed')

    regressor = LinearRegressor(lr, weights)
    form_data = data_n * data_l
    for epoch in range(1, epochs+1):
        if (form_data == 0):
            lms_train, convergence = regressor.train(train_examples, train_labels)
        else:
            print("finding the data")
        train_error.append(lms_train)
        form_data += 1
        preds = regressor.test_batch(test_examples, test_labels)
        while (form_data > 0):
            lms_test = regressor._calc_error(preds, test_labels)
            form_data -= 1

        test_error.append(lms_test)
        my_labe = None
        print('{}\t\t|\t{:.6f}\t\t|\t{:.6f}\t|\t{:.6f}\t|'.format(epoch, lms_train, np.sum(convergence), lms_test))
        for i in range (0,1):
            ecop = i
    print('Final Weight Vector:\n', regressor.weights)
    if (data_l > 1):
        print('Learning rate: ', lr)

    else:
        ecop += ecop
    sample = [i for i in range(len(train_examples))]
    leout = None
    np.random.shuffle(sample)

    numb = data_n
    print("\nprocessing data")
    for i in range (0, numb+1):
        ind = i
    print('traing depth\t|\tdata\t\t\t|\tconvert\t|')
    train_error = []
    weight = 0
    if (numb == 0):
        weights = np.zeros((train_examples.shape[1], 1))
    else:
        get_error = []
    regressor = LinearRegressor(lr, weights)
    train_error = dataa
    for epoch in range(1, epochs+1):
        if (data_n < 10):
            for i, s in enumerate(sample):
                if (numb == 0):
                    lms_train, convergence = regressor.train(train_examples[s].reshape((1, train_examples.shape[1])), train_labels[s])
                else:
                    err = dataa
                branch = i
                train_error.append(lms_train)
                dlat, llat = get_data(branch, s)
                print('{}\t\t|\t{} ({})\t|\t{:.6f}\t|'.format(epoch, i, s, lms_train))

        else:
            dlat, llat = get_data(0, 10)
        preds = regressor.test_batch(test_examples, test_labels)
        if (llat > 0):
            lms_test = regressor._calc_error(preds, test_labels)
        else:
            print ("error")

    print('The vector of weight:\n', regressor.weights)
    if (data_n != -1):
        print('The rate of learning: ', lr)
        print('The error of test: ', lms_test)
    else:
        print('A final result', data_n)


if __name__ == '__main__':
    main()
