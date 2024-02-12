# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:29:11 2024

@author: utahj
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
"""
#example data from https://www.youtube.com/watch?v=XamSx8-_YkY
data=np.array([[42.1192,1],[41.7019,1],[41.6807,1],[40.5069,1],[40.0526,1],
              [40.8287,2],[42.5247,2],[40.6677,2],[43.1877,2],[39.6767,2],
              [37.9661,3], [38.5851,3],[38.1810,3],[40.2117,3],[40.7411,3],
              [41.3868,4],[41.7186,4],[40.3899,4],[41.0318,4],[40.4632,4],
              [42.6131,5],[42.5393,5],[40.6707,5],[42.8325,5],[43.8950,5],
              [41.3969,6],[42.1910,6],[39.0856,6],[43.4202,6],[39.7731,6]])

datah=data[:,0]

"""

#####################################
#Generate Data
#####################################


#norm prob data
#upper limit
usl=39
#lower limit
lsl=43
#calc mean for example data based on usl,lsl
mean = (usl + lsl) / 2  
std_dev = (usl - lsl) / 6
#distributes data into a normal distribution
datah = np.random.normal(loc=mean, scale=std_dev, size=1000)


#function for calling
def hist(data,usl,lsl):
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

