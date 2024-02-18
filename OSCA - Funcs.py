# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:29:11 2024

@author: utahj
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
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
    [10.05, 10.05, 10.01, 10.00, 10.04]
])

#norm prob data
#upper limit
usl=10.05
#lower limit
lsl=9.95
#calc mean for example data based on usl,lsl
mean = (usl + lsl) / 2  
std_dev = (usl - lsl) / 6


#function for calling
def hist(data,usl,lsl):
    #############Histogram
    #plots histogram
    data=data.flatten()
    #plt.hist(data, bins=10, density=True, alpha=0.6, color='blue', edgecolor='black')
    #calculates mean and std dev of data input
    mu, sigma = np.mean(data), np.std(data)
    #plots the USL and LSL lines on the plot
    #plt.axvline(x=usl, color='red', linestyle='--', linewidth=2)
    #plt.text(usl,plt.gca().get_ylim()[1], f'USL= {usl}', fontsize=12, color='r')
    #plt.axvline(x=lsl, color='red', linestyle='--', linewidth=2)
    #plt.text(lsl,plt.gca().get_ylim()[1], f'LSL= {lsl}', fontsize=12, color='r')

    xmin, xmax = np.min(data),np.max(data)
    xhist = np.linspace(xmin, xmax, 100)
    phist = norm.pdf(xhist, mu, sigma)
    #plt.plot(xhist, phist, 'k', linewidth=2)
    #plt.xlabel('Value')
    #plt.ylabel('Frequency')
    #plt.show()
    return xhist, phist

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
    #plt.plot(sx,means,marker='o')
    #c=1
    #to further distinguish subgroups that are out of control, we will indicate these as a red dot
    #for i in means:
        #if i>=usl or i<=lsl:
            #plt.plot(c,i,marker='o',color='r')
        #c+=1
        
    #the next following lines are for adding text onto the plot to distinguish the line meanings.
    #plt.ylim(lsl-std_dev,usl+std_dev)
    #plt.axhline(y=usl,color='r')
    #plt.axhline(y=lsl,color='r')
    #plt.text(sn+1.2,usl-.002, f'UCL= {usl}',color='r', fontsize=12)
    #plt.text(sn+1.2,lsl-.002, f'LCL= {lsl}',color='r', fontsize=12)
    #plt.axhline(y=xbarr,color='g')
    #plt.text(sn+1.2,xbarr-.002, r'$\overline{\overline{x}}$' + f' = {np.round(xbarr,3)}', fontsize=12)
    #plt.ylabel('Sample Mean',fontsize=12)
    #plt.title('Xbar Chart',fontsize=14)
    return sn,ss,sx,mean

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
    elif ss==3:
        a2=1.023
        d3=0
        d4=2.574
    elif ss==4:
        a2=.729
        d3=0
        d4=2.282
    elif ss==5:
        a2=.577
        d3=0
        d4=2.004
    elif ss==6:
        a2=.483
        d3=0
        d4=2.004
    elif ss==7:
        a2=.419
        d3=.076
        d4=1.924
    elif ss==8:
        a2=.373
        d3=.136
        d4=1.864
    elif ss==9:
        a2=.337
        d3=.184
        d4=1.816
    elif ss==10:
        a2=.308
        d3=.223
        d4=1.777
    elif ss==15:
        a2=.223
        d3=.347
        d4=1.653
    elif ss==25:
        a2=.153
        d3=.459
        d4=1.541
    #now we can calculate the ucl and lcl for the R chart
    rbar=np.sum(ranges)/np.size(ranges)
    uclr=d4*rbar
    lclr=d3*rbar
    #plt.plot(sx,ranges,marker='o')
    #plt.ylim(-.01,np.max(ranges)+std_devm)
    #plt.xticks(sx,sx)
    #plt.axhline(uclr,color='r')
    #plt.axhline(lclr,color='r')
    #plt.axhline(rbar,color='g')
    c=1
    #to further distinguish subgroups that are out of control, we will indicate these as a red dot
    #for i in ranges:
        #if i>=uclr or i<=lclr:
            #plt.plot(c,i,marker='o',color='r')
        #c+=1
    #plt.title('R Chart',fontsize=14)
    #plt.ylabel('Sample Range',fontsize=12)
    return ranges, uclr, lclr

#Subgroup Plotting
#we just need to plot a scatter for this one.
def last(data):
    sn,ss=np.shape(data)
    sx=np.linspace(1,sn,sn)
    sx=sx.astype(int)
    c=0
    #while c<sn:
        #for i in data[c,:]:
            #plt.scatter(c+1,i,color='b',marker='+')
        #c+=1
        #if c==sn:
            #break
    #plt.xticks(sx)
    #plt.yticks([np.round(np.mean(data)-std_dev*3,2),np.round(np.mean(data)+std_dev*3,2)])
    #plt.ylim(np.min(data-std_dev*3),np.max(data+std_dev*3))
    #plt.axhline(np.mean(data),linestyle='--',alpha=.5)
    #plt.gca().set_aspect(40)
    #plt.title(f'Last {sn}' +' Subgroups',fontsize=14)
    #plt.ylabel('Values',fontsize=12)
    #plt.xlabel('Sample',fontsize=12)

#Probability Plot
#each probability is calculated indexwise.
#first sort data
def prob(data):
    sn,ss=np.shape(data)
    sx=np.linspace(1,sn,sn)
    sx=sx.astype(int)

    fdata = data.flatten()
    sdata = np.sort(fdata)
    pr = (np.arange(len(sdata)) + 0.5) / len(sdata)
    flat_data = data.flatten()
    result = stats.anderson(flat_data, dist='norm')
    ad = result.statistic

    #plt.xlim(np.min(sdata)-std_dev/2,np.max(sdata+std_dev/2))
    #plt.ylim(0,np.max(pr))
    lr=np.polyfit(sdata,pr,1)
    lrdata=lr[0]*sdata+lr[1]

    fit=np.polyfit(sdata,pr,3)
    fdata = fit[0] * sdata**3 + fit[1] * sdata**2 + fit[2]*sdata+fit[3]
    #plt.plot(sdata,fdata,color='g',alpha=.4,linewidth=3)

    #plt.grid(axis='x', which='major', linestyle='')
    #plt.plot(sdata,lrdata,color='r',alpha=.5)
    #plt.scatter(sdata, pr,marker='o',facecolors='none',edgecolors='blue')
    #plt.yticks([])
    #plt.title('Normal Probability Plot')
    #plt.grid(True)
    statistic, p = stats.shapiro(data)

    #plt.text(np.max(sdata)+std_dev*5/4, .5, f'Mean: {np.round(np.mean(data),3)} \nStandard Deviation: {np.round(np.std(data),3)} \nN: {len(data)}\nAD:{np.round(ad,3)}\nP-Value:{np.round(p,3)}', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    return sdata,pr,fdata,lrdata,p

def osca(data,usl,lsl):
    hist(data,usl,lsl)
    xbar(data)
    rbar(data)
    last(data)
    prob(data)

osca(data,usl,lsl)



