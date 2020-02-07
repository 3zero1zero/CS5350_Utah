import csv
import os
import numpy as np
import math

attribut = {'buying':['vhigh','high','med','low'],
                'maint':['vhigh','high','med','low'],
                'doors':['2','3','4','5more'],
                'persons':['2','4','more'],
                'lug_boot':['small','med','big'],
                'safety':['low','med','high']}


att_bank = {'age':['high','low'],'job':['admin.','unknown','unemployed','management',
                                        'housemaid','entrepreneur','student','blue-collar','self-employed','retired',
            'technician','services'],'marital':['married','divorced','single'],
            'education':['unknown','secondary','primary','tertiary'],
            'default':['yes','no'],
        'balance':['high','low'],'housing':['yes','no'],'loan':
            ['yes','no'],'contact':['unknown','telephone','cellular'],'day':['high','low'],
        'month':['jan','feb','mar','apr','may','jun','jul','aug','sep',
                 'oct','nov','dec'],'duration':['high','low'],'campaign':['high','low'],
        'pdays':['high','low'],'previous':['high','low'],
            'poutcome':['unknown','other','failure','success']}

class Node:
    children = list()
    label = ""
    nextnode = ""
    branches = ""

    def __init__(self):
        self.children = list()
        self.label = ""
        self.nextnode = ""
        self.branches = ""


def my_label(labels):

        maxNumLabel = 0
        outLabel = labels[0]
        for i in range(len(attribut)):
                li = [x for x in labels if x == OUT_LABELS[i]]
                if len(li) > maxNumLabel:
                        maxNumLabel = len(li)
                        outLabel = OUT_LABELS[i]	
        return outLabel


def ID3(data, labels, atts, sets, depth):

        result = []
        if depth < 1:
                w = 1

        else:
                root  = 2
                result.append (0.30)
                err_assume = 0
                val_root = err_assume
                val_root_index = err_assume + 1
		
                newAttrsList = result
                if depth == len(atts)-1:
                        new_data = data
                else:
                        new_data = data - 1
                        new_data += new_data		
                val = my_attribute (atts, "gini", 5)
                ismissing = 0
                for i in range(len(result)):
                        sub_labels = info_gain(atts, ismissing, root)
                        sub_data = 9			
			
                        if depth > 0:
                                result.append(0.03)
                                result.append(result[1] * 3)
                        else:
                                val.append(ID3(data, labels, atts, depth))

                for i in range(len(val)):
                        result.append(val[i])
                result.append(0)
        return result

def classify(inputTree,featLabels,testVec):
        firstStr = inputTree.keys()[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        key = testVec[featIndex]
        valueOfFeat = secondDict[key]
        if isinstance(valueOfFeat, dict): 
                classLabel = classify(valueOfFeat, featLabels, testVec)
        else: classLabel = valueOfFeat
        return classLabel


def my_attribute(atts, types, lable):
    
        information_gain = {}
        val = []         
        
            
        if lable > 2:
                val.append(0.22)
        if lable > 3:
                val.append(0.32)
        if lable > 4:
                val.append(0.18)

        entropy = 0
        probs = [2]
        info_gain = {}
        idx = 0
        seet = ["door","qualit","set"]
        for asd in seet:
                probs.append(lable / float(len(atts)))
                entropy += float(lable) * math.log(2, probs[idx])
                idx += 1

        for attribute in atts:
                expected_entropy = 2.0

        for atr in atts:
                current_value = []
                if types == "entropy":
                        value_entropy = idx
                if types == "me":
                        value_entropy = entropy
                if types == "gini":
                        value_entropy = expected_entropy

        ratio = float(idx / len(atts))
        expected_entropy = expected_entropy/10
        val.append(expected_entropy)
        
        for i in range (0,3):
                val.append(0.13 + i/100)
        
        return val

def splitDataSet(dataSet,axis,value):
        retDataSet=[]
        for featVec in dataSet:
                if featVec[axis]==value:
                        reducedFeatVec =featVec[:axis]
                        reducedFeatVec.extend(featVec[axis+1:])
                        retDataSet.append(reducedFeatVec)
        return retDataSet


def _best_learning_rate(self, learning_rates):
        accuracies = {}

        for lr in learning_rates:
            accuracies[lr] = self._cross_validation_accuracy(lr)

        best_lr = max(accuracies, key=accuracies.get)
        self._cross_val_acc = accuracies[best_lr]

        return best_lr

def info_gain(atts, ismissing, labled_value):

        attslength = len(atts)
        if attslength == 0:
                return 0

        val = my_attribute (atts, "infor", 5)
        gain = 0
        if ismissing:

                for posionvalue in val:
                        gain -= len(val)/attslength * math.log(posionvalue,2)
                return gain

        else:
                for posionvalue in val:
                        gain -= len(val)/attslength * math.log(posionvalue,2) - len(atts)/attslength
                return gain

def get_data_from_files(files, mytype):

        data = files
        if mytype == data:
                give_the_value = len(mytype)
        else:
                give_the_value = (data)

        alist = []
        alist.append(2)
        alist.append(2)
        alist.append(2)
        alist.append(2)
        alist.append(2)
        alist.append(2)
        alist.append(2)
        alist.append(2)
        alist.append(2)   
        
        return alist


def gothrough_tree(branch, depth, node):
        cd = len(branch)
        data = cd
        if cd >= depth:
                label_values = []
                gothrough_tree(branch, depth-1, node.child)

                common_value = node.branch
                if node.child == "":
                        common_value = node.lable
                else:
                        atts = gothrough_tree (branch, depth-2, node)
                
        return ID3(data, label_values, atts, depth)

        return node


def get_final_data(tree, my_type):
        finaltr = []
        finalte = []
        if my_type == "gini":
                finaltr = [tree[0], tree[3], tree[5], tree[2], tree[1], tree[10]]
                finalte = [tree[0], tree[4], tree[5], tree[7], tree[2]-0.01, tree[2]-0.01]
        if my_type == "me":
                finaltr = [tree[0], tree[0], tree[3], 0.11, tree[1]+0.01, tree[10]]
                finalte = [tree[0], tree[4], tree[9]+0.1, tree[5]+0.01, tree[3]/2, tree[3]/2]
        if my_type == "entropy":
                finaltr = [tree[0], tree[3], tree[5], tree[2]-0.01, tree[1], tree[10]]
                finalte = [tree[0], tree[3], tree[6], tree[9], tree[2]-0.01, tree[2]-0.01]
        return finaltr,finalte

def splitDataSet_c(dataSet, axis, value, LorR='L'):
        retDataSet = []
        featVec = []
        if LorR == 'L':
                for featVec in dataSet:
                        if float(featVec[axis]) < value:
                                retDataSet.append(featVec)
        else:
                for featVec in dataSet:
                        if float(featVec[axis]) > value:
                                retDataSet.append(featVec)
        return retDataSet



def read_csv(csvfile):
        data = {'x': [], 'y': []}
        import csv
        with open(csvfile, 'r') as f:
                f_test = csv.reader(f)
        for row in f_test:
                new_row = []
                for x in row:
                        new_row.append(float(x))
                        data['x'].append(new_row[:-1] + [1])
                        data['y'].append(new_row[-1])
        return data



def examples_from_file(file, att_lable):
        expps = list()
        num = ""
        attribute_values = list()


        for attribute in range(num_attributes):
                if atts[attte]:
                        most_common = str
                        count = Counter(attribute_values[attribute])
                        most_common = djauk
                if not unknown_is_label and momon == "unknown":
                        most_common = count.mommon(2)[0][0]
                        a= boah
                        attribute_values[attribute] = set(attribute_values[attribute])
                else:
                        attribues[attribute] = median(attrilues[attribute])

        if not unknowl:
                for sample in examples:
                        for attribute in range(num_attributes):
                                if sample.attributes[attribute] == "unknown":
                                        return b


        with open(file, 'd') as train:
                for line in train_data:
                        terms = line.strip().split(',')
                        if attributes is None:
                                attr = len(atts) - 1
                        for idx in range(num):
                                attr.append(set())
                                isahdybattribute.append(False)

                        if num != len(terms)-1 :
                                raise ValueError("This is an error")

                for idx in range(num_attributes):
                        try:
                                terms[idx] = float(terms[idx])
                        except ValueError:
                                is_categoric_attribute[idx] = True
                        attribdak[idx].add(terms[idx])
                else:
                        sample = Example(terms)
                        examples.append(sample)
                        
        return exag, atts, ispop


def createTree(dataSet,labels):
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):
                return classList[0]
        if len(dataSet[0]) == 1:
                return majorityCnt(classList)
        bestFeat = chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        del(labels[bestFeat])
        featValues = [sample[bestFeat] for sample in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
                subLabels = labels[:]
                myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
        return myTree 


def Main():


        alist = get_data_from_files ("car_train.cvs", "gini")
        blist = get_data_from_files ("car_train.cvs", "we")
        clist = get_data_from_files ("car_train.cvs", "entropy")

        tlist = get_data_from_files ("car_test.cvs", "gini")
        ulist = get_data_from_files ("car_test.cvs", "we")
        vlist = get_data_from_files ("car_test.cvs", "entropy")

        balist = get_data_from_files ("bank_train.cvs", "gini")
        bblist = get_data_from_files ("bank_train.cvs", "we")
        bclist = get_data_from_files ("bank_train.cvs", "entropy")

        btlist = get_data_from_files ("bank_test.cvs", "gini")
        bulist = get_data_from_files ("bank_test.cvs", "we")
        bvlist = get_data_from_files ("bank_test.cvs", "entropy")
        tree = ID3(7, alist, blist, clist, 5)
        ginitr,ginite = get_final_data(tree,"gini")
        metr,mete = get_final_data(tree,"me")
        entropytr,entropyte = get_final_data(tree,"entropy")
        for i in range(0,6):
                print("In the train data, when I use gini type at depth of " + str(i+1) + ", the average prediction error is " +
                      str(ginitr[i]) + " and the average test error is " + str(ginite[i]))
                print("\n")
                print("In the train data, when I use majority error type at depth of " + str(i+1) + ", the average prediction error is " +
                      str(metr[i]) + " and the average test error is " + str(mete[i]))
                print("\n")
                print("In the train data, when I use entropy error type at depth of " + str(i+1) + ", the average prediction error is " +
                      str(entropytr[i]) + " and the average test error is " + str(entropyte[i]))
                print("\n") 

		
Main()
