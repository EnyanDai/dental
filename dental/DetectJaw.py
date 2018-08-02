# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:07:51 2018

@author: Enyan
"""
import numpy as np
from glm import rescale_img
import cv2
from landmarks import landmarks

class Detector:
    def __init__(self,images,number_of_components = 9):
        self.shape = images[0].shape
        mat = np.asarray(images).reshape((-1,self.shape[0]*self.shape[1]))
        self.eigenvalues,self.eigenvectors = self.pca(mat,number_of_components=number_of_components)
        
    def pca(self,X,number_of_components):
        self.mean = np.mean(X,axis=0)
        L = np.subtract(X,self.mean)
        Lt =np.transpose(L)
        corv = np.matmul(L,Lt)
        eigenvalues,eigenvectors= np.linalg.eigh(corv)
        eigenvectors = np.matmul(Lt,eigenvectors)
        eigenvectors = np.divide(eigenvectors,np.linalg.norm(eigenvectors,axis=0))
        eigenvalues = eigenvalues[X.shape[0]-number_of_components:X.shape[0]]
        eigenvectors = eigenvectors[:,X.shape[0]-number_of_components:X.shape[0]]
        
        self.b = L@eigenvectors
        
        return eigenvalues,eigenvectors
        
    def reconstruct(self,x):
        x = x.reshape((-1))
        L = np.subtract(x,self.mean)
        projection = np.matmul(L,self.eigenvectors)
        loss = self.distance(projection)
        
        return loss

    def distance(self,projection):
        distance = np.subtract(self.b,projection)
        distance = np.sum(distance*distance,axis = 1)
        distance = np.min(distance)
        return distance
        
    def findInBox(self,test):
        w = self.shape[1]
        h = self.shape[0]
        nx = test.shape[1] - w + 1
        ny = test.shape[0] - h + 1
        losses = np.zeros((nx,ny))
        for i in range(nx):
            for j in range(ny):
                img = test[j:j+h,i:i+w]
                losses[i,j] = self.reconstruct(img)
        ind = np.unravel_index(np.argmin(losses, axis=None), losses.shape)
        return ind

class DetectJaw(Detector):
    
    def find_box(self,test,px,py):
        x = int(px*test.shape[0])
        y = min(int(py*test.shape[0]),test.shape[0]-self.shape[0])
        left = int((test.shape[1]-self.shape[1])/2)
        right = left+self.shape[1]
        roi = test[x:y,left:right]
        ind = self.findInBox(roi)
        top = x

        rescale_w = 1.1
        rescale_h = 1.1
        roi_w = int(rescale_w*self.shape[1])
        roi_h = int(rescale_h*self.shape[0])
        left = max(left - int((roi_w - self.shape[1])/2),0)
        top = max(top + ind[1] - int((roi_h - self.shape[0])/2),0)
        
        roi = test[top:top+roi_h,left:left+roi_w]
        ind = self.findInBox(roi)
        top_left = (ind[0]+left,ind[1]+top)
        buttom_right=(top_left[0]+self.shape[1],top_left[1]+self.shape[0])
        
        return top_left,buttom_right
class DetectTooth(Detector):
    
    def find_tooth(self,test,top_left,buttom_right,number):
        number = number%4
        width = int((buttom_right[0] - top_left[0])/4)
        left = top_left[0]+number*width
        top = top_left[1]

        rescale_w = 1.1
        rescale_h = 1.1
        roi_w = int(rescale_w*self.shape[1])
        roi_h = int(rescale_h*self.shape[0])
        
        left = max(left - int((roi_w - self.shape[1])/2),0)
        top = max(top - int((roi_h - self.shape[0])/2),0)
        print(top_left,buttom_right)
        print(left,top,roi_w,roi_h)
        print(self.shape)
        roi = test[top:top+roi_h,left:left+roi_w]
        ind = self.findInBox(roi)
        tooth_top_left = (ind[0]+left,ind[1]+top)
        tooth_buttom_right=(tooth_top_left[0]+self.shape[1],tooth_top_left[1]+self.shape[0])
        
        return tooth_top_left,tooth_buttom_right
        
    

#for img in teeth[]:
#    cv2.imshow('show',img)
#    k = cv2.waitKey(0)

class AllDetector:
    def __init__(self,data,scale,n=10):
        self.scale = scale
#        n = 10
        up,down = data.get_box(n)
        teeth = data.get_all_teeth(n)
        self.UpperJawDetector = self.buildJawDetector(up,scale,6)
        self.LowerJawDetector = self.buildJawDetector(down,scale,6)
        self.TeethDector = []
        for imageset in teeth:
            self.TeethDector.append(self.buildToothDetector(imageset,scale,9))
            
    def buildJawDetector(self,imageset,scale,number_of_components):
        imageset = rescale_img(imageset,scale)
        return DetectJaw(imageset,number_of_components)
    def buildToothDetector(self,imageset,scale,number_of_components):
        imageset = rescale_img(imageset,scale)
        return DetectTooth(imageset,number_of_components)
        
    def detect_teeth(self,img,show = False):
        position = []
        test = cv2.resize(img,(int(self.scale*img.shape[1]),int(self.scale*img.shape[0])))
        up_top_left,up_buttom_right = self.UpperJawDetector.find_box(test,0.35,0.7)
        down_top_left,down_buttom_right = self.LowerJawDetector.find_box(test,0.5,1)
        for n,tooth in enumerate(self.TeethDector[0:4]):
            tooth_top_left,tooth_buttom_right = tooth.find_tooth(test,up_top_left,up_buttom_right,n)
            position.append((tooth_top_left,tooth_buttom_right))
        for n,tooth in enumerate(self.TeethDector[4:]):
            tooth_top_left,tooth_buttom_right = tooth.find_tooth(test,down_top_left,down_buttom_right,n)
            position.append((tooth_top_left,tooth_buttom_right))
        position.append((up_top_left,up_buttom_right))
        position.append((down_top_left,down_buttom_right))
        if(show):
            cv2.rectangle(test,up_top_left,up_buttom_right,(255),1)
            cv2.rectangle(test,down_top_left,down_buttom_right,(255),1)
            for posi in position[0:8]:
                cv2.rectangle(test,posi[0],posi[1],(255,0,0),1)
            cv2.imshow('Detect jaw',test)
            cv2.waitKey(0)
        return self.scale_up_posi(position),test
    def scale_up_posi(self,position):
        posi = []
        for (top_left,buttom_right) in position:
            lx = int(top_left[0]/self.scale)
            ly = int(top_left[1]/self.scale)
            rx = int(buttom_right[0]/self.scale)
            ry = int(buttom_right[1]/self.scale)
            posi.append(((lx,ly),(rx,ry)))
        return posi

         
#alldetector = AllDetector(data,1,11)
#img = data.images[11]
#alldetector.detect_teeth(img,show=True)
#cv2.destroyAllWindows()



