# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:29:11 2024

@author: utahj
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from matplotlib.ticker import MultipleLocator
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
    [9.96, 10.01, 9.98, 10.03, 10.02],
    [10.02, 9.98, 10.00, 10.01, 9.99],
    [9.96, 9.97, 10.01, 10.02, 10.03],
    [10.01, 10.00, 9.98, 10.04, 9.99],
    [10.05, 9.98, 10.03, 9.97, 10.01],
    [10.03, 10.05, 9.98, 10.00, 10.02],
    [10.01, 9.99, 10.03, 9.95, 10.00],
    [10.08, 10.01, 10.02, 9.99, 9.97],
    [10.02, 10.03, 10.01, 9.98, 9.99],
    [9.97, 9.95, 10.01, 9.96, 10.02],
    [9.99, 10.00, 9.98, 10.03, 10.01],
    [10.08, 9.99, 10.02, 9.98, 10.00],
    [10.01, 10.00, 9.96, 9.96, 10.02],
    [10.12, 9.99, 10.03, 9.97, 10.01],
    [10.05, 10.01, 10.00, 10.02, 10.03],
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
    return xhist, phist, sigma, mu

def xbar(data):
    #first we need to assess the amount of subgroups when given random data from user input.
    #sn is subgroup number, ss is subgroup size.
    sn,ss=np.shape(data)
    #for each subgroup we need to collect the sample mean in the group.
    means=np.mean(data,axis=1)
    stdw=np.std(data,axis=1)
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
    return sn,ss,sx,mean,stdw,means,xbarr

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
    rbarr=np.sum(ranges)/np.size(ranges)
    uclr=d4*rbarr
    lclr=d3*rbarr
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
    return ranges, uclr, lclr, std_devm, rbarr

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
    flatdata = data.flatten()
    result = stats.anderson(flatdata, dist='norm')
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
    return sdata,pr,fdata,lrdata,p,flatdata,ad

def osca(data,usl,lsl):
    xhist, phist, sigma, mu=hist(data,usl,lsl)
    sn,ss,sx,mean,stdw,means,xbarr=xbar(data)
    ranges, uclr, lclr, std_devm, rbarr =rbar(data)
    last(data)
    sdata,pr,fdata,lrdata,p,flatdata,ad=prob(data)
    wsigma=np.mean(stdw)
    cp=(usl-lsl)/(6*wsigma)
    cpk=np.min([(usl-mu),(mu-lsl)])/(3*wsigma)
    dp=0
    for i in flatdata:
        if i < lsl or i > usl:
            dp+=1
    dpu=dp/len(flatdata)
    ppm=dp/len(flatdata)*1000000
    pp=(usl-lsl)/(6*std_dev)
    if np.abs(mu-usl)>np.abs(mu-lsl):
        ppk=(mu-lsl)/(3*std_dev)
    else:
        ppk=(usl-mu)/(3*std_dev)
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0,0].plot(xhist, phist, 'k', linewidth=2)
    axs[0,0].set_xlabel('Value')
    axs[0,0].set_ylabel('Frequency')
    axs[0,0].axvline(x=usl,color='r')
    axs[0,0].axvline(x=lsl,color='r')
    axs[0,0].hist(flatdata,bins=12,edgecolor='black',color='blue')
    axs[0,0].set_title('Capability Histogram',fontsize=14)
    axs[0, 0].text(lsl+std_dev/6, np.max(flatdata)*2.5, 'LCL', fontsize=12, color='red')
    axs[0, 0].text(usl+std_dev/6, np.max(flatdata)*2.5, 'UCL', fontsize=12, color='red')
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    axs[2, 1].axis('off')
    
    axs[0,1].plot(sx,means,marker='o')
    axs[0,1].axhline(y=xbarr,color='g')
    axs[0,1].set_ylim(lsl-std_dev,usl+std_dev)
    axs[0,1].axhline(y=usl,color='r')
    axs[0,1].axhline(y=lsl,color='r')
    axs[0,1].text(sn+1.2,usl-.002, f'UCL= {usl}',color='r', fontsize=12)
    axs[0,1].text(sn+1.2,xbarr-.002, r'$\overline{\overline{x}}$' + f' = {np.round(xbarr,3)}', fontsize=12)
    axs[0,1].text(sn+1.2,lsl-.002, f'LCL= {lsl}',color='r', fontsize=12)
    axs[0,1].set_title('Xbar Chart',fontsize=14)
    c=1
    for i in means:
        if i>=usl or i<=lsl:
            axs[0,1].plot(c,i,marker='o',color='r')
        c+=1
    
    
    axs[1,1].plot(sx,ranges,marker='o')
    axs[1,1].set_ylim(-.01,np.max(ranges)+std_devm)
    axs[1,1].set_xticks(sx.astype(int),sx.astype(int))
    xmin = min(sx.astype(int))
    xmax = max(sx.astype(int))
    tick_positions = np.linspace(xmin, xmax, num=len(sx)*2-1, endpoint=True)
    tick_labels = [label if i % 2 == 0 else '' for i, label in enumerate(sx.astype(int))]
    axs[1, 1].set_xticklabels(tick_labels)
    

    axs[1, 1].tick_params(axis='x', pad=10)  # Adjust the value of 'pad' as needed
    
    axs[1,1].text(sn+1.2,uclr-.002, f'UCL= {usl}',color='r', fontsize=12)
    axs[1,1].text(sn+1.2,lclr-.002, f'LCL= {lsl}',color='r', fontsize=12)
    axs[1,1].axhline(y=uclr,color='r')
    axs[1,1].axhline(y=lclr,color='r')
    axs[1,1].axhline(y=rbarr,color='g')
    c=1
    for i in ranges:
        if i>=uclr or i<=lclr:
            axs[1,1].plot(c,i,marker='o',color='r')
        c+=1
    axs[1,1].set_title('R Chart',fontsize=14)
    axs[1,1].set_ylabel('Sample Range',fontsize=12)
    
    
    c=0
    while c<sn:
        for i in data[c,:]:
            axs[1,0].scatter(c+1,i,color='b',marker='+')
        c+=1
        if c==sn:
            break
    axs[1,0].set_xticks(sx)
    axs[1,0].set_yticks([np.round(np.mean(data)-std_dev*3,2),np.round(np.mean(data)+std_dev*3,2)])
    axs[1,0].set_ylim(np.min(data-std_dev*3),np.max(data+std_dev*3))
    axs[1,0].axhline(np.mean(data),linestyle='--',alpha=.5)
    axs[1,0].set_aspect(40)
    axs[1,0].set_title(f'Last {sn}' +' Subgroups',fontsize=14)
    axs[1,0].set_ylabel('Values',fontsize=12)
    axs[1,0].set_xlabel('Sample',fontsize=12)
    axs[1, 0].xaxis.set_major_locator(MultipleLocator(5))
    
    
    
    axs[2,0].set_xlim(np.min(sdata)-std_dev/2,np.max(sdata+std_dev/2))
    axs[2,0].set_ylim(0,np.max(pr)+np.max(pr)/2)
    lr=np.polyfit(sdata,pr,1)
    #axs[2,0].plot(sdata,fdata,color='g',alpha=.4,linewidth=3)

    axs[2,0].grid(axis='x', which='major', linestyle='')
    axs[2,0].plot(sdata,lrdata,color='r',alpha=.5)
    axs[2,0].scatter(sdata, pr,marker='o',facecolors='none',edgecolors='blue')
    axs[2,0].set_yticks([])
    axs[2,0].set_title('Normal Probability Plot')
    axs[2,0].grid(True)

    axs[2,0].text(np.max(sdata)+std_dev*5/4, .35, f'Mean: {np.round(np.mean(data),3)} \nStandard Dev: {np.round(np.std(data),3)} \nN: {np.size(data)}\nAD:{np.round(ad,3)}\nP-Value:{np.round(p,3)}', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    

    
    
    
    axs[0, 0].text(-.25, 1.4, 'Open Source Capability Analysis (OSCA) Report', transform=axs[0, 0].transAxes, ha='left', va='center', fontsize=20,fontstyle='italic')


    
    
    
    
    text_box = axs[2, 1].text(2, .6, 'Process Metrics \n' + f'Standard Deviation= {np.round(std_dev,3)}\n' + f'Cp= {np.round(cp,3)}\n' + f'Cpk= {np.round(cpk,3)}\n'+ f'Pp= {np.round(pp,3)}\n'+ f'Ppk= {np.round(ppk,3)}\n'+ f'Ppm= {np.round(ppm,3)}',
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=axs[2, 0].transAxes,
                           bbox=dict(facecolor='whitesmoke', edgecolor='black', boxstyle='round,pad=1'))




    plt.show()

    
    

osca(data,usl,lsl)



