# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:44:11 2018

@author: Enyan
"""
import numpy as np
import matplotlib.pyplot as plt

def transform(mark,t):
    '''
    cent should be in (2,)
    '''
    mark = mark.reshape((-1,2))
    return np.add(mark,t).reshape((-1))
def centralized(mark):
    mark = mark.reshape((-1,2))
    centroid = np.mean(mark,axis=0)
    mark = np.subtract(mark,centroid)
    return mark.reshape((-1))
def get_centroid(mark):
    mark = mark.reshape((-1,2))
    centroid = np.mean(mark,axis=0)
    return centroid
def plot_mark(mark):
    mark = mark.reshape((-1,2))
    plt.plot(mark[:,0],mark[:,1],'r')
    ax = plt.gca()
    plt.axis("equal")
    ax.invert_yaxis()
    plt.show()
    
def normalized(mark):
    mark = mark.reshape((-1))
    norm = np.linalg.norm(mark)
    return mark/norm

def get_alignment_parameters(x1,x2):
#==============================================================================
#     This method is to rotate x1 by (s,theta) to minimise |sAx1 - x2|
#==============================================================================
    x1 = x1.reshape((-1))
    x2 = x2.reshape((-1))
    a = (np.dot(x1,x2))/(np.linalg.norm(x1)**2)
    x1 = x1.reshape((-1,2))
    x2 = x2.reshape((-1,2))
    b = ( np.dot(x1[:,0],x2[:,1]) - np.dot(x1[:,1],x2[:,0]) )/(np.linalg.norm(x1)**2)
    
    s = np.sqrt(a**2 + b**2)
    
    theta = np.arctan(b/a)
    
    return s,theta
def scale_and_rotation(x1,s,theta):
    x1 = x1.reshape((-1,2))
    transform = np.zeros((2,2))
    transform[0,0] = s*np.cos(theta)
    transform[0,1] = s*np.sin(theta)
    transform[1,0] = -s*np.sin(theta)
    transform[1,1] = s*np.cos(theta)
    x1 = np.matmul(x1,transform)
    return x1.reshape((-1))
def align(x1,x2):
    '''
    x1 and x2 both should be centralized
    x2 is the target vector
    x1 is the vector need to be aligned
    '''
    s,theta = get_alignment_parameters(x1,x2)
    x1 = x1.reshape((-1,2))
    x1 = scale_and_rotation(x1,s,theta)
    
    norm = np.dot(x1,x2.reshape(-1))
    
    return x1/norm

def GPA(marks):
    '''
    The marks should be in N X M martirx 
    N is the number of samples
    M is 2*number of points in a landmark
    '''
    TableMarks = []
    for i in range(marks.shape[0]):
        TableMarks.append(centralized(marks[i,:]))
    x0 = TableMarks[0]
    x0 = normalized(x0)
    mean_shape = x0
    
    while True:
        for i,vector in enumerate(TableMarks):
            TableMarks[i] = align(vector,mean_shape)
        mean_shape2 = np.mean(TableMarks,axis = 0)
        mean_shape2 = align(mean_shape2,x0)
        mean_shape2 = normalized(mean_shape2)
        change = np.sum(np.absolute(mean_shape2-mean_shape))
#        print(change)
        mean_shape = mean_shape2
        if( change < 1e-12):
            break
        
    return TableMarks,mean_shape
    
#    while True:

#mark = data.marks[:11,0,:]
#mark = mark.reshape((-1,80))
#TableMarks,mean_shape = GPA(mark)
#plot_mark(mean_shape)
    