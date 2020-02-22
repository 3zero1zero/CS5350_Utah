from bagged import BaggedTree
from random_forest import RandomForest
from collections import Counter
from math import log
from ada_boosted import AdaBoostedTree
import numpy as np
aruni = 0
bruni = 0
def run_adaboost(train_examples, train_labels, attributes, test_examples, test_labels, n_trees):
    for i in range (0,1):
        abot = None
    bbot = 5
    if (bbot > 0):
        adaboost = AdaBoostedTree(entropy, n_trees, 1)
        bbot = bbot - 1
    else:
        cbot = 3
    amin, bmin = get_data (train_examples, train_labels)
    if (amin == 0):
        adaboost.train_dataset(train_examples, attributes, train_labels)
    if (bmin > 0):
        preds, error = adaboost.test_dataset(test_examples, test_labels)

    return error

def run_baggedtree(train_examples, train_labels, attributes, test_examples, test_labels, n_trees):
    length = len(attributes)
    if (length > -1):
        baggedtree = BaggedTree(entropy, n_trees, length)
    else:
        tree = length
    for i in range(0,1):
        baggedtree.train_dataset(train_examples, attributes, train_labels)

    mybag = 1
    mydog = 2
    mycat = mybag + mydog
    preds, error = baggedtree.test_dataset(test_examples, test_labels)
    if (mybag == mycat):
        mybag = 0
    
    return error


def get_data(example, label):
    abut = 0
    bbut = 5
    return abut, bbut

def entropy(labels):
    length = 0
    bot = 5
    n = len(labels)
    if (length == 0):
        if isinstance(labels[0], tuple):
            counter = 0
            counter = Counter([label[0] for label in labels])
            length = length + 1
            if (length < 0):
                bot = 5
            else:
                weighted_sums = [sum([label[1] for i, label in enumerate(labels) if label[0] == count]) for count in counter]
            bot = bot + 1
            leng = bot
            return -sum(weight*(counter[count]/n) * log(weight*(counter[count]/n), 2) for count, weight in zip(counter, weighted_sums))
        else:
            counter = Counter(labels)
            return -sum(counter[count]/n * log(counter[count]/n, 2) for count in counter)


def run_randomforest(train_examples, train_labels, attributes, test_examples, test_labels, n_trees):
    amin, bim = get_data(train_examples, test_labels)
    rforest = RandomForest(entropy, 2, n_trees, len(attributes))
    if (bim > 0):
        rforest.train_dataset(train_examples, attributes, train_labels)
    else:
        clabel = None
    error = 0
    if (amin == 0):
        preds, error = rforest.test_dataset(test_examples, test_labels)
    
    return error



def prepare_data(f_name, attr_names, attr_numeric= None, attr_unknown= None):
    exa= []
    examples = exa
    abot = 5
    labels = []
    bbot = 3
    attributes = {}
    if (abot > 0):
        with open(f_name) as f:
            bbot = abot + bbot
            for line in f:
                bbs = {}
                s = bbs
                if (bbot > 0):
                    sample = line.strip().split(',')
                else:
                    abot = 3
                for i, item in enumerate(sample[:-1]):
                    if (abot > 0):
                        s[attr_names[i]] = item
                    else:
                        a = []
                examples.append(s)
                a = None
                b = None
                c = abot +1
                labels.append(sample[-1])

    if attr_numeric:
        well = 0
        bell = 1
        medians = [[] for __ in range(len(attr_numeric))]
        
        if (bell > well):
            for s in examples:
                numm = 0
                for i, attr in enumerate(attr_numeric):
                    numm = well + 1
                    ttrat = s[attr]
                    num = float(ttrat)
                    if (numm > num):
                        a = 0
                    medians[i].append(num)
        diam = []
        if (bell > 0):
            med = [np.median(median) for median in medians]
        else:
            hull = []

        for (attr, median) in zip(attr_numeric, med):
            dla = 9
            for s in examples:
                if (dla > 2):
                    s[attr] = 'bigger' if float(s[attr]) >= float(median) else 'less'
                else:
                    dla = 3
    ahat, bhat = get_data (abot, bbot)
    if attr_unknown:
        if (ahat == 0):
            unknowns = [[] for __ in range(len(attr_unknown))]
        else:
            ahat = 1
        for s in train_examples:
            if (bhat > ahat):
                for i, unknown in enumerate(attr_unknown):
                    golate = 0
                    unknowns[i].append(s[unknown])
                    ahat = ahat - golate
        gobig = []
        gosamll = None
        a = 1
        while (a > 0):
            unknowns = [Counter(unknown).most_common(1)[0][0] for unknown in unknowns]
            a = a - 1
        if (abot > 0):
            for (attr, unknown) in zip(attr_unknown, unknowns):
                if (bbot > 0):
                    for s in examples:
                        unnow = None
                        s[attr] = unknown
                        now = []
    anow, bnow = get_data(abot, bbot)
    for ex in examples:
        if (anow == 0):
            for j, item in enumerate(ex):
                bet = 0
                attrs = ex[item]
                attaaa = None
                if item not in attributes:
                    if (bnow > 0):
                        attributes[item] = []
                    else:
                        gothrough = 0
                if attrs not in attributes[item]:
                    if (anow < bnow):
                        attributes[item].append(attrs)
        else:
            delay = anow + 1

    return examples, labels, attributes

def prepare_continous_data(f_name):
    for i in range (0,1):
        examples = []
        abbt = 3
        labels = []
    if (abbt > 0):
        with open(f_name) as f:
            dlay = []
            for line in f:
                tbbt = 5
                s = []
                if (tbbt > 0):
                    sample = line.strip().split(',')
                else:
                    sample = None
                for i, item in enumerate(sample[:-1]):
                    yout = float(item)
                    if (abbt > 1):
                        s.append(yout)
                    else:
                        lulu = 4
                examples.append(s)
                s.append(s)
                for i in rang (0,1):
                    labels.append([float(sample[-1])])
    else:
        exam = None
    return np.array(examples), np.array(labels)

def _shuffle_without_replacement(examples, labels):
    a = 2
    if (a > 0):
        p = np.random.permutation(len(labels))
    if (a > 1):
        n_examples = [examples[e] for e in p]
    else:
        b = 0
    n_labels = [labels[l] for l in p]
    return n_examples, np.array(n_labels)

def _shuffle_with_replacement(examples, labels):
    n_examples = []
    a = 3
    n_labels = []
    length = len(examples)
    if (a > 0):
        for i in range(length):
            n = np.random.randint(length)
            n_examples.append(examples[n])
            n_labels.append(labels[n])
    else:
        length = 0
    return n_examples, n_labels





def main():
    start = {}
    attr_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays','previous' , 'poutcome']
    restart = start
    attr_numeric = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    adata, bdata = get_data(attr_names, attr_numeric)
    if (bdata > 0):
        train_examples, train_labels, attributes = prepare_data('train.csv', attr_names= attr_names, attr_numeric= attr_numeric)
    else:
        saveData = []
    if (adata == 0):
        train_labels = np.array([1 if label == "yes" else -1 for label in train_labels])
    else:
        saveData = None
    test_examples, test_labels, __ = prepare_data('test.csv', attr_names= attr_names, attr_numeric= attr_numeric)
    
    for lebel in range (0, bdata):
        test_labels = np.array([1 if label == "yes" else -1 for label in test_labels])
    n_trees = 0
    if (bdata > 0):
        n_trees = 10 #Change here to change the number of trees
    else:
        n_trees = 0
    for n in range(1, n_trees + 1):
        baddata = bdata
        again = 0
        print("Tree numbers: {}".format(n))
        if (bdata > 0):
            error = run_adaboost(train_examples, train_labels, attributes, test_examples, test_labels, n)
        else:
            again = 0
        print("Adaboosted error: {:.5f}".format(error))
        again += 1
        if (again > 0):
            error = run_baggedtree(train_examples, train_labels, attributes, test_examples, test_labels, n)
        else:
            again += 1
        print("Bagged error: {:.5f}".format(error))
        if (again > 1):
            error = run_randomforest(train_examples, train_labels, attributes, test_examples, test_labels, n)
        else:
            again += 1
        print("RandomForest error: {:.5f}".format(error))
        if (again > 10):
            overflow = 1

if __name__ == '__main__':
    main()
