#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:24:07 2017

@author: hari
"""

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
'''
objects = ('DART', 'RF', 'GBDT')
y_pos = np.arange(len(objects))
performance = [0.53,0.58,0.61]
 
plt.bar(y_pos, performance, align='center', alpha=0.8, width=0.4)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Boosting Methods vs Accuracy')
plt.savefig('Boosting vs accuracy.jpg') 
plt.show()

objects = [0.12,0.31,0.01]
#y_pos = np.arange(len(objects))
performance = [ 0.3701,0.3757,0.38,0.3839,0.3867,0.3886,0.3897,0.3905,0.3906,0.3903,
    0.3892,0.3877,0.3857,0.3834,0.3808,0.3779,0.3746,0.371,0.3669]
 
plt.bar(objects, performance)
plt.xticks(y_pos, objects)
plt.ylabel('F1')
plt.xlabel('Threshold')
plt.title('Boosting Methods vs Accuracy')
#plt.savefig('Boosting vs accuracy.jpg') 
plt.show()

'''
X=np.arange(0.12,0.31,0.01)
Y2 = np.empty(19)
Y2.fill(6.31)
Y1=[ 0.3701,0.3757,0.38,0.3839,0.3867,0.3886,0.3897,0.3905,0.3906,0.3903,
    0.3892,0.3877,0.3857,0.3834,0.3808,0.3779,0.3746,0.371,0.3669]
#Y3=[ 15.45,14.29,13.26,12.34,11.51,10.76,10.09,9.47,8.91,8.39,7.92,7.49,
#    7.08,6.7,6.35,6.03,5.72,5.43,5.16]

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
#lns1 = ax.plot(X, Y2, '-', label = 'Actual')
#lns2 = ax.plot(X, Y3, '-', label = 'Predicted')
#ax2 = ax.twinx()
#lns3 = ax2.plot(X, Y1, '-r', label = 'F1')
lns3 = ax.plot(X, Y1, '-b', label = 'F1')
#lns = lns1+lns2+lns3
#labs = [l.get_label() for l in lns]
#ax.legend(lns, labs, loc=0)
ax.set_xlabel('Threshold')
ax.set_ylabel('F1')
#ax2.set_ylabel('F1')
plt.suptitle('F1 vs Threshold', size=12)
plt.savefig('F1_vs_mean_cart_size.jpg')
plt.show()