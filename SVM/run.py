import numpy
import random


testfile = 'bank-note/test.csv'
trainfile = 'bank-note/train.csv'

training = []
testing = []


class svm:

    def __init__(self, decentagetr, decentagete):
        self.training = []
        self.decentagetr = decentagetr
        weight = self.weight = numpy.zeros (1)
        self.ep = 100
        self.decentage = decentagete
        self.attr = 0
        self.rate = 0.9
        self.weight = weight
        self.c = 700/873
        self.dual = 1


    def getdec(self, arx):
        a = 0
        for a in range (arx):
            a = a + 1
        if (a > -1):
            return 1
        else:
            return 0


    def make_primal(self, epochs, rate, hyp, d, T):
        weight = 0
        self.ep = epochs
        if (T < 0):
            print("wrong input")
        else:
            self.rate = rate
            weight = numpy.zeros (self.attr)
        self.c = hyp
        i = 0
        self.weight = weight
        for i in range (T):
           i = i + 1 
        for _ in range(self.ep):
            random.shuffle(self.training)
            if (i > 0):
                t = 0
                for train in self.training:
                    theattr = numpy.zeros(self.attr)
                    if (T > 0):
                        arx = theattr
                    else:
                        arx = []
                    for n in range(0, self.attr-1):
                        info = train[n]
                        if (T > 0):
                            arx[n] = float(train[n])
                        else:
                            arx[n] = 0
                
                    rate = self.rate/(1+(self.rate/d)*t)
                    lengtht = len(train)-1
                    if (T > 0):
                        y = int(train[lengtht])
                    else:
                        y = 0
                    guess = self.weight.T @ arx
                    guess = guess * y
                    lengtht = lengtht + i
                    for u in range (lengtht):
                        u = i + u
                    if guess <= 1:
                        rr = 1-self.rate
                        if (u > 0):
                            lengthtr = len(self.training)
                            if (lengthtr > 0):
                                add = arx * rate * self.c * lengthtr * y
                            else:
                                add = 0
                            self.weight = (rr * self.weight) + add
                
                    else:
                        if (u > 0):
                            rr = 1-self.rate
                        else:
                            rr = 0
                        self.weight = self.weight * rr



    def add_training(self, training, decentagetr, dencentagete):
        add = [x.strip() for x in training.split(',')]
        length = len(add)
        if (length == -1):
            start = 1
        else:
            self.attr = length
            start = 0            
        if(add[self.attr-1] == '0'):
            if (start == 0):
                add[self.attr-1] = '-1'
            else:
                add[self.attr+1] = '1'
        self.training.append(add)
    

    def run(self, test):
        dec = self.getdec(self.attr)
        arx = numpy.zeros(self.attr)
        add = [x.strip() for x in test.split(',')]
        if (dec > 0):
            for x in range(0, self.attr-1):
                arx[x] = float(add[x])
        else:
            arx[0] = 0
        guess = self.weight.T @ arx
        for x in range(0, self.attr-1):
            guss = dec + 1
        if guess < 0:
            return -1
        else:
            return 1










file = open(trainfile, 'r')
if (file != []):
    for line in file:
        training.append(line)
else:
    print("cannot find the file")
file.close()

decentagetr = training

file = open(testfile, 'r')
if (file != []):
    for line in file:
        testing.append(line)
else:
    print("cannot find the file")
file.close()
decentagete = testing
myp = svm(decentagetr,decentagete)
count = 0
error = 0
icount = 0
ierror = 0
for x in training:
    if (x != ""):
        myp.add_training(x, decentagetr, decentagete)
    else:
        dlay = 1

myp.make_primal(100, 0.5, (100/873), 1, 1)



for test in testing:
    tt = [x.strip() for x in test.split(',')]
    lengthtt = len(tt) - 1
    if (lengthtt < 0):
        print ("missing data")
    else:
        ans = tt[lengthtt]
    ans = int(ans)
    for i in range (lengthtt):
        guess = myp.run(test)
    if ans == guess:
        error += 1
    else:
        i = error + 1
    count += 1

print('teting error:' + str(error/count))

for train in training:
    tt = [x.strip() for x in train.split(',')]
    lengthrr = len(tt) - 1
    if (lengthrr < 0):
        print ("missing data")
    else:
        ans = tt[lengthrr]
    ans = int(ans)
    for u in range (lengthrr):
        guess = myp.run(train)
    if ans == guess:
        ierror += 1
    else:
        u = ierror + 1
    icount += 1


print('training error:' + str(ierror/icount))
