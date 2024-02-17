# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:29:11 2024

@author: utahj
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

#####################################
#Generated Data
#####################################

"""
20 subgroups of 5 data points each.

"""

data = np.array([
    [9.95, 9.97, 9.96, 10.02, 9.98],
    [10.03, 10.01, 9.99, 10.00, 10.04],
    [10.05, 9.96, 9.94, 10.03, 10.01],
    [10.02, 10.07, 10.00, 10.01, 9.99],
    [10.00, 10.01, 9.98, 10.03, 10.02],
    [10.02, 9.98, 10.00, 10.01, 9.99],
    [9.96, 9.97, 10.01, 10.02, 10.03],
    [10.01, 10.00, 9.98, 10.04, 9.99],
    [10.00, 9.98, 10.03, 9.97, 10.01],
    [10.03, 10.05, 9.98, 10.00, 10.02],
    [10.01, 9.99, 10.03, 9.95, 10.00],
    [10.00, 10.01, 10.02, 9.99, 9.97],
    [10.02, 10.03, 10.01, 9.98, 9.99],
    [9.97, 9.95, 10.01, 10.00, 10.02],
    [9.99, 10.00, 9.98, 10.03, 10.01],
    [10.01, 9.99, 10.02, 9.98, 10.00],
    [10.01, 10.00, 9.96, 10.01, 10.02],
    [10.02, 9.99, 10.03, 9.97, 10.01],
    [10.00, 10.01, 10.00, 10.02, 10.03],
    [9.97, 10.02, 10.01, 10.00, 10.04]
])

#norm prob data
#upper limit
usl=10.05
#lower limit
lsl=9.95
#calc mean for example data based on usl,lsl
mean = (usl + lsl) / 2  
std_dev = (usl - lsl) / 6
#distributes data into a normal distribution
datah = np.random.normal(loc=mean, scale=std_dev, size=1000)

#function for calling
def hist(datah,usl,lsl):
    #############Histogram
    #plots histogram
    plt.hist(datah, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
    #calculates mean and std dev of data input
    mu, sigma = np.mean(datah), np.std(datah)
    #plots the USL and LSL lines on the plot
    plt.axvline(x=usl, color='red', linestyle='--', linewidth=2)
    plt.text(usl,plt.gca().get_ylim()[1], f'USL= {usl}', fontsize=12, color='r')
    plt.axvline(x=lsl, color='red', linestyle='--', linewidth=2)
    plt.text(lsl,plt.gca().get_ylim()[1], f'LSL= {lsl}', fontsize=12, color='r')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    



#hist(datah,usl,lsl)


#####################################
#Xbar
#####################################

def xbar(data):
    #first we need to assess the amount of subgroups when given random data from user input.
    #sn is subgroup number, ss is subgroup size.
    sn,ss=np.shape(data)
    #for each subgroup we need to collect the sample mean in the group.
    means=np.mean(data,axis=1)
    #now we have means for each subgroup as the Y and the subgroup number as the X.
    #we need a vector to represent the subgroup number across the x axis.
    sx=np.linspace(1,sn,sn)
    #we also need to calculate the grand average, or the center line.
    xbarr=np.sum(data)/(np.size(data))
    #we will then plot all of the values onto the Xbar chart
    plt.plot(sx,means,marker='o')
    c=1
    #to further distinguish subgroups that are out of control, we will indicate these as a red dot
    for i in means:
        if i>=usl or i<=lsl:
            plt.plot(c,i,marker='o',color='r')
        c+=1
        
    #the next following lines are for adding text onto the plot to distinguish the line meanings.
    plt.ylim(lsl-std_dev,usl+std_dev)
    plt.axhline(y=usl,color='r')
    plt.axhline(y=lsl,color='r')
    plt.text(sn+1.2,usl-.002, f'UCL= {usl}',color='r', fontsize=12)
    plt.text(sn+1.2,lsl-.002, f'LCL= {lsl}',color='r', fontsize=12)
    plt.axhline(y=xbarr,color='g')
    plt.text(sn+1.2,xbarr-.002, r'$\overline{\overline{x}}$' + f' = {np.round(xbarr,3)}', fontsize=12)
    plt.ylabel('Sample Mean',fontsize=12)
    plt.title('Xbar Chart',fontsize=14)
    return sn,ss,sx
    
## R Bar Chart
#data has already been characterized
#now ranges will be calculated for each subgroup.
def rbar(data):
    sn,ss=np.shape(data)
    sx=np.linspace(1,sn,sn)
    sx=sx.astype(int)
    ranges = [np.ptp(row) for row in data]
    std_devm=np.std(ranges)
    #we also need to calculate the correct control limits.
    #to do so, we need to incorporate the usage of control constants based upon subgroup size.
    #we already know subgroup size.
    if ss==2:
        a2=1.88
        d3=0
        d4-3.267
    if ss==3:
        a2=1.023
        d3=0
        d4=2.574
    if ss==4:
        a2=.729
        d3=0
        d4=2.282
    if ss==5:
        a2=.577
        d3=0
        d4=2.004
    if ss==6:
        a2=.483
        d3=0
        d4=2.004
    if ss==7:
        a2=.419
        d3=.076
        d4=1.924
    if ss==8:
        a2=.373
        d3=.136
        d4=1.864
    if ss==9:
        a2=.337
        d3=.184
        d4=1.816
    if ss==10:
        a2=.308
        d3=.223
        d4=1.777
    if ss==15:
        a2=.223
        d3=.347
        d4=1.653
    if ss==25:
        a2=.153
        d3=.459
        d4=1.541
    #now we can calculate the ucl and lcl for the R chart
    rbar=np.sum(ranges)/np.size(ranges)
    uclr=d4*rbar
    lclr=d3*rbar
    plt.plot(sx,ranges,marker='o')
    plt.ylim(-.01,np.max(ranges)+std_devm)
    plt.xticks(sx,sx)
    plt.axhline(uclr,color='r')
    plt.axhline(lclr,color='r')
    plt.axhline(rbar,color='g')
    c=1
    #to further distinguish subgroups that are out of control, we will indicate these as a red dot
    for i in ranges:
        if i>=uclr or i<=lclr:
            plt.plot(c,i,marker='o',color='r')
        c+=1
    plt.title('R Chart',fontsize=14)
    plt.ylabel('Sample Range',fontsize=12)

#Subgroup Plotting
#we just need to plot a scatter for this one.
def last(data):
    sn,ss=np.shape(data)
    sx=np.linspace(1,sn,sn)
    sx=sx.astype(int)
    c=0
    while c<sn:
        for i in data[c,:]:
            plt.scatter(c+1,i,color='b',marker='+')
        c+=1
        if c==sn:
            break
    plt.xticks(sx)
    plt.yticks([np.round(np.mean(data)-std_dev*3,2),np.round(np.mean(data)+std_dev*3,2)])
    plt.ylim(np.min(data-std_dev*3),np.max(data+std_dev*3))
    plt.axhline(np.mean(data),linestyle='--',alpha=.5)
    plt.gca().set_aspect(40)
    plt.title(f'Last {sn}' +' Subgroups',fontsize=14)
    plt.ylabel('Values',fontsize=12)
    plt.xlabel('Sample',fontsize=12)

#Probability Plot
#each probability is calculated indexwise.
#first sort data
sn,ss=np.shape(data)
sx=np.linspace(1,sn,sn)
sx=sx.astype(int)


fdata = data.flatten()
sdata = np.sort(fdata)
pr = (np.arange(len(sdata)) + 0.5) / len(sdata)

    
plt.scatter(sdata, pr)
plt.ylabel('Ordered Values')
plt.title('Normal Probability Plot')
plt.grid(True)
plt.show()



