# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 23:03:03 2018

@author: Enyan
"""
import numpy as np
import procrustes_analysis as pa
class PCA:
    
    def __init__(self,marks,accuracy):
        
        self.mean = np.mean(marks,axis = 0)
        self.eigenvector,self.eigenvalue = self.getValidEigenvectors(marks,accuracy)
        
    def getValidEigenvectors(self,marks,accuracy):
        
        mean = np.mean(marks,axis = 0)
        s = marks.shape[0]
        central = np.subtract(marks,mean)
        covr = np.matmul(central.transpose(),central)/s
        eigenvalue, eigenvector = np.linalg.eigh(covr)
        eigenvalue = eigenvalue[::-1]
        eigenvector = eigenvector[::-1,:]
        total_var = np.sum(eigenvalue)
        parameters = []
        var = 0
        
        for i in range(len(eigenvalue)):
            var += eigenvalue[i]
            parameters.append(eigenvector[i,:])
            if(var/total_var >= accuracy):
                break
            
        return np.asarray(parameters),eigenvalue[0:i+1]
        
    def Reconstruct(self,mark):
        
        mark = mark.reshape((-1))
        mark = mark-self.mean
        reconstruction = mark@self.eigenvector.transpose()@(self.eigenvector)
        reconstruction = reconstruction + self.mean
        
        return reconstruction
        
    def addconstraint(self,x):
        x = x - self.mean
        b = x@self.eigenvector.transpose()
#        print(b)
        b = np.clip(b,-3*np.sqrt(self.eigenvalue),3*np.sqrt(self.eigenvalue))
#        print(b)        
                
        reconstruction = b@self.eigenvector+self.mean
        
        return reconstruction
        
    def addPoints(self,mark):
        x = self.mean
        mark = mark.reshape((-1,2))
        cent =  pa.get_centroid(mark)
        mark = pa.centralized(mark)
        
        for i in range(50):
            
            s,theta = pa.get_alignment_parameters(mark,x)
            y = pa.scale_and_rotation(mark,s,theta)
            y = y/np.dot(y,self.mean)
            
            new_x = self.Reconstruct(y)
            change = np.sum(np.absolute(new_x-x))
#            print(change)
            x = new_x
            if(change < 1e-12):
                break;
                
        s,theta = pa.get_alignment_parameters(x,mark)
        x = self.addconstraint(x)
        
        reconstruct_mark = pa.scale_and_rotation(x,s,theta)
        reconstruct_mark = np.add(reconstruct_mark.reshape(-1,2),cent)
        
        return reconstruct_mark.reshape((-1))
        


    