# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 19:50:49 2018

@author: Enyan
"""
from glm import MultiResASM
from DetectJaw import AllDetector
from landmarks import landmarks
class AllAsm:
    def __init__(self,data,n):
        self.detector = AllDetector(data,1,n)
        self.multiAsm = self.buildAllAsm(data,n)
    def buildAllAsm(self,data,n):
        multiAsm = []
        train_images = data.images[0:n]
        marks = data.marks[0:n,:,:]
        for i in range(8):
            mark = marks[:,i,:].reshape((-1,80))
            multiAsm.append(MultiResASM(train_images,mark))
        return multiAsm
    def get_marks(self,img):
        posi,test = self.detector.detect_teeth(img,show=True)
#        print(posi)
        TableMarks = []
        for i,asm in enumerate(self.multiAsm):
            print(posi[i])
            TableMarks.append(self.multiAsm[i].fit(posi[i],img))
        return TableMarks            
                
landmarksPath = "C:\\Users\\Enyan\\Desktop\\cv\\_Data\\Landmarks\\original\\"
ImagePath = "C:\\Users\\Enyan\\Desktop\\cv\\_Data\\Radiographs\\"
data = landmarks(ImagePath,landmarksPath)
n = 11
model = AllAsm(data,n)
img = data.images[11]
model.get_marks(img)

