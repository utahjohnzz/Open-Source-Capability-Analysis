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
    [9.7, 10.00, 9.96, 10.01, 10.02],
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
    sx=np.linspace(1,20,20)
    #we also need to calculate the grand average, or the center line.
    xbcl=np.sum(data)/(np.size(data))
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
    plt.axhline(y=xbcl,color='g')
    plt.text(sn+1.2,xbcl-.002, r'$\overline{\overline{x}}$' + f' = {np.round(xbcl,3)}', fontsize=12)
    plt.ylabel('Sample Mean',fontsize=12)
    plt.title('Xbar Chart',fontsize=14)
    



