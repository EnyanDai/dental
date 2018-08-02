# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:34:57 2018

@author: Enyan
"""

from PCA import PCA
from procrustes_analysis import GPA,plot_mark
import numpy as np
for i in range(8):
    mark = data.marks[:11,i,:]
    mark = mark.reshape((-1,80))
    TableMarks,mean_shape = GPA(mark)
    plot_mark(mean_shape)

marks = np.asarray(TableMarks)
accuracy = 0.98
PCAmodel = PCA(marks,accuracy)
for t in range(-3,4):
    new_shape = PCAmodel.mean+t*np.sqrt(PCAmodel.eigenvalue[1])*PCAmodel.eigenvector[1]
    plot_mark(new_shape)