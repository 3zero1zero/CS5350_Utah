import glob
import os
import librosa
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram


def secret (clip):
    d1 = []
    d2 = []
    data = []
    use = []
    startp = []
    endp = []
    slop = []
    long = []
    length = len(clip)
    for i in range (length - 1):
        d1.append(clip[i])
        d2.append(clip[i+1])
        if (d2[i] > d1[i]):
            if (i == 0):
                data.append(0)
            else:
                data.append(data[i-1] + 1)
        if (d2[i] < d1[i]):
            if (i == 0):
                data.append(0)
            else:
                data.append(data[i-1] - 1)
        if (d2[i] == d1[i]):
            if (i == 0):
                data.append(0)
            else:
                data.append(data[i-1])
        use.append(data[i])

    ignore_y = (max(use)-min(use)) * 0.1
    ignore_x = len(use)* 0.02


    long.append (ignore_y)

    hold = 0
    current = 0
    direct = 0
    predir = 0
    l = -1
    precurrent = 0
    hi = 0
    prei = 0

    for i in range (len(use)):  
        current = use[i]
        if (len(use) - i < 2):
            l = i - l
            rate = ((current + precurrent)/2 - hold) / l
            hold = (current + precurrent)/2
            #print (rate)
            slop.append(rate * 1000)
            startp.append(prei)
            endp.append(i)
            #print (hold)
            #print (i)

        else:
            if (current - precurrent > ignore_y and (i-prei) > ignore_x):
                direct = 1
            
                if (predir != direct):
                    startp.append(prei)
                    prei = i
                    if (predir == -1):
                        l = i - l
                        rate = ((current + precurrent)/2 - hold) / l
                        hold = (current + precurrent)/2
                        predir = 1
                        hi = i
                    if (predir == 0):
                        l = i - l
                        rate = precurrent / l
                        hold = precurrent
                        predir = 1
                        hi = i
            
                    precurrent = hold
                   # print (rate)
                   # print (hold)
                   # print (i)
                    slop.append(rate * 1000)
                    endp.append(i)
                else:
                    precurrent = current
                    hi = i

        
            if (current - precurrent < (ignore_y*-1) and (i-prei) > ignore_x):
                direct = -1
                if (predir != direct):
                    startp.append(prei)
                    prei = i
                    if (predir == 1):
                        l = i - l
                        rate = ((current + precurrent)/2 - hold) / l
                        hold = (current + precurrent)/2
                        predir = -1
                        hi = i
                    if (predir == 0):
                        l = i - l
                        rate = precurrent / l
                        hold = precurrent
                        predir = 1
                        hi = i
            
                    precurrent = hold
                    #print (rate)
                    #print (hold)
                    #print (i)
                    slop.append(rate*1000)
                    endp.append(i)
                else:
                    precurrent = current
                    hi = i
        
            if ((ignore_y * -1) < current - precurrent < ignore_y):
                #direct = 0
                if (i - hi > ignore_x and direct != predir):
                    startp.append(prei)
                    direct = 0
                    prei = i
                    l = i - l
                    rate = (use[hi] - hold) / l
                    hold = use[hi]
                    predir = 0
                    precurrent = current
                    #precurrent = hold
                    slop.append(rate * 1000)
                    endp.append(i)
                    #print (rate)
                    #print (hold)
                    #print (i)
                
    return startp,endp,slop,long



clips = []
context = []
for i in range (1,36):
    clip, sample_rate = librosa.load("c"+str(i)+".wav", sr=None)
    clips.append(clip)
    
for y in range (35):
    st, en, sl, lo = secret (clips[y])
    for z in range (len(st)):
        if (0 <= y <=4):
            row = [st[z],en[z],sl[z],lo[0],1]
        if (5 <= y <=10):
            row = [st[z],en[z],sl[z],lo[0],2]
        if (11 <= y <=16):
            row = [st[z],en[z],sl[z],lo[0],3]
        if (17 <= y <=19):
            row = [st[z],en[z],sl[z],lo[0],4]
        if (20 <= y <=25):
            row = [st[z],en[z],sl[z],lo[0],5]
        if (26 <= y <=30):
            row = [st[z],en[z],sl[z],lo[0],6]
        if (31 <= y <=34):
            row = [st[z],en[z],sl[z],lo[0],7] 
        context.append(row)


print (context)

filename = "secret_data.csv"

with open(filename, 'w', newline = '') as csvfile:  

    csvwriter = csv.writer(csvfile)  
         
    csvwriter.writerows(context)



