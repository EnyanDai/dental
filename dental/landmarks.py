# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 18:58:36 2018

@author: Enyan
"""
import cv2
import os
import numpy as np
import re

class landmarks():
    
    def __init__(self,ImagePath,landmarksPath):
        self.images = self.LoadImages(ImagePath)
        self.marks = self.LoadLandmarks(landmarksPath)
        
    def LoadImages(self,Path):
#==============================================================================
#         load the image from path
#==============================================================================
        images = []
        for i in sorted(os.listdir(Path)):
            if i.endswith('.tif'):
                path = Path + i
                images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
        return images
    def LoadLandmarks(self,landmarksPath):
#==============================================================================
#         The format of the landmarks is 
#         (x1,y1,x2,y2......,x39,y39,x40,40)
#==============================================================================
        marks = np.zeros([14,8,2*40])
        pattern = re.compile('[\d]+-[\d]+')
        for i in os.listdir(landmarksPath):
            result = pattern.search(i)
            if(result!=None):
                tmp = np.loadtxt(landmarksPath+i,dtype=float)
                [img,number] = result.group(0).split("-")
                img = int(img)-1
                number = int(number)-1
                marks[img,number,:]=tmp[:]
        return marks        
            
    def get_box(self,n):
        a = self.marks.reshape((-1,4*40,2))
        a = np.asarray(a[:2*n,:,:],dtype = int) 
        xmin = np.min(a[:,:,0],axis = 1).reshape((-1,2))
        xmax = np.max(a[:,:,0],axis = 1).reshape((-1,2))
        ymin = np.min(a[:,:,1],axis = 1).reshape((-1,2))
        ymax = np.max(a[:,:,1],axis = 1).reshape((-1,2))
        
        width = xmax-xmin
        height = ymax-ymin
        
        w = np.asarray(width.mean(axis = 0),dtype=int)
        h = np.asarray(height.mean(axis = 0),dtype=int)
        up = []
        down = []

        for i in range(n):
            img = self.images[i]
            roi_up = img[ymin[i,0]:ymax[i,0],xmin[i,0]:xmax[i,0]]
            roi_up = cv2.resize(roi_up,(w[0],h[0]))
            up.append(roi_up)

            roi_down = img[ymin[i,1]:ymax[i,1],xmin[i,1]:xmax[i,1]]
            roi_down = cv2.resize(roi_down,(w[1],h[1]))
            down.append(roi_down)

        return up,down
        
    def get_all_teeth(self,n):
        number = 8
        a = self.marks.reshape((-1,40,2))
        a = np.asarray(a[:number*n,:,:],dtype = int) 
        xmin = np.min(a[:,:,0],axis = 1).reshape((-1,number))
        xmax = np.max(a[:,:,0],axis = 1).reshape((-1,number))
        ymin = np.min(a[:,:,1],axis = 1).reshape((-1,number))
        ymax = np.max(a[:,:,1],axis = 1).reshape((-1,number))
        
        width = xmax-xmin
        height = ymax-ymin
        
        w = np.asarray(width.mean(axis = 0),dtype=int)
        h = np.asarray(height.mean(axis = 0),dtype=int)
        teeth = []
        for i in range(number):
            tooth = []
            teeth.append(tooth)
            
        for i in range(n):
            img = self.images[i]
            for j in range(number):
                roi = img[ymin[i,j]:ymax[i,j],xmin[i,j]:xmax[i,j]]
                roi = cv2.resize(roi,(w[j],h[j]))
                teeth[j].append(roi)
        return teeth

def plotLandmarks(image,mark,thickness = 1):
    '''
    mark can be int or float in any () format
    '''
    mark = np.round(mark)
    mark = mark.reshape((-1,80))
    landmarks = np.asarray(mark,dtype=int)
    
    dst = np.zeros_like(image)
    dst[:]=image[:]
    for j in range(mark.shape[0]):
        points = landmarks[j,:]
        points = points.reshape((-1,1,2))
        dst=cv2.polylines(dst,[points],True,(255),thickness = thickness)
    return dst
        


        
        
