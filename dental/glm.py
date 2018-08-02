# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 23:30:29 2018

@author: Enyan
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
from landmarks import plotLandmarks
import procrustes_analysis as pa
from PCA import PCA

def rescale_img(images,scale):
    new_scale_imgs=[]
    for img in images:
        new_shape = (int(scale*img.shape[1]),int(scale*img.shape[0]))
        new_scale_imgs.append(cv2.resize(img,new_shape))
    return new_scale_imgs

def rescale_marks(marks,scale):
    new_scale_marks = np.round(scale*marks)
    return new_scale_marks
    
def getnormal(dv):
    p = np.zeros(2)
    p[0] = -dv[1]
    p[1] = dv[0]
    norm = np.linalg.norm(p);
    if(norm == 0.0):
        p[0] = 1
    else:
        p = p/np.linalg.norm(p)
    return np.round(p).astype(int)
def normalise_g(g):
    '''
    g should be in (N,k)
    '''
    norm_g = np.zeros_like(g)
    norm = np.sum(np.absolute(g),axis = 1)
#    print(g.shape)
#    print(norm.shape)
    for i in range(norm.shape[0]):
#        print(norm[i])
        norm_g[i] = g[i]/norm[i]
    return norm_g
        
def getprofile(mark,k,sobelx,sobely,n = 40):
    
    mark = mark.reshape(n,2)
    profile = np.zeros((n,2*k+1,2),dtype = int)
    normal = np.zeros((n,2))
    grad = np.zeros((n,2*k+1),dtype = float)
    
    for i in range(n):
        v0 = mark[(i-1)%n,:]
        v2 = mark[(i+1)%n,:]
        v1 = mark[i,:]
        p = getnormal(v2-v0)
        normal[i,:] = p[:]
        for j in range(2*k+1):
            posi = (j-k)*p+v1
            
            posi = np.round(posi).astype(int)
            if(posi[1] < -100):
                print(v0)
                print(v1)
            profile[i,j,:] = posi[:]
            g = np.zeros(2)
            g[0] = sobelx[posi[1],posi[0]]
            g[1] = sobely[posi[1],posi[0]]
            grad[i,j] = np.dot(g,p)/np.linalg.norm(p)
    return profile,normal,grad
    
def getg(img,marks,k,normalise = True):
    '''
    The marks will be reshaped
    '''
    marks = marks.reshape((-1,80))
    n = marks.shape[0]
    sobelx = cv2.Sobel(img,cv2.CV_64F,dx = 1,dy = 0, ksize = 3)
    sobely = cv2.Sobel(img,cv2.CV_64F,dx = 0,dy = 1, ksize = 3)
    grad = np.zeros((marks.shape[0]*40,2*k+1))
    profile = np.zeros((marks.shape[0]*40,2*k+1,2),dtype = int)
    for i in range(n):
        p,normal,g = getprofile(marks[i],k,sobelx,sobely)
        profile[40*i:40*(i+1),:] = p
        grad[40*i:40*(i+1),:] = g
    if(normalise):
        grad = normalise_g(grad)
    return profile,grad
    
class glm:
    ''' gray level model
    '''
    def __init__(self,images,marks,m = 3):
        '''
        the marks should be in (N , K , D )
        N is the number of samples
        K is the number of teeth
        D is the 2 * number of points in a landmark
        '''
        self.m = m
        print('build glm')
        n = len(images)
        marks = marks.astype(float).reshape((n,-1,80))
        self.grads= self.build(images,marks,m)
        self.g_mean = self.grads.mean(axis = 0)
        inv_cov = []
        for i in range(self.grads.shape[1]):
#            print(i)
            inv_cov.append(np.linalg.inv(np.cov(self.grads[:,i,:].reshape((len(images),-1)),rowvar = False)))
        self.inv_cov = inv_cov
        
    def build(self,images,marks,m):
        import procrustes_analysis as pa
        grads = []
        for i,img in enumerate(images):
            profile,grad = getg(img,marks[i],m)
            grads.append(grad)
#            pa.plot_mark(profile)
        grads = np.asarray(grads)
        return grads
        
    def maha_distance(self,gs,gmean,inv_cov):
        gs = gs/np.sum(np.absolute(gs))
        dis = (gs-gmean)@inv_cov@(gs-gmean).transpose()
        return dis.reshape((-1))
        
    def find_best(self,img,mark,k = 7):
        '''
        return the mark in (N,2) which N is total point in the mark
        '''
        profile,grad = getg(img,mark,k,normalise=False)
        new_marks = np.zeros((grad.shape[0],2),dtype = int)
        p = 0
        for j in range(grad.shape[0]):
            MaDistance = np.zeros(2*(k-self.m) + 1)
            for i in range(2*(k-self.m) + 1):
                g = grad[j,i:i+2*self.m+1]
                MaDistance[i] = self.maha_distance(g,self.g_mean[j],self.inv_cov[j])
            ns = np.argmin(MaDistance)
            if(abs(ns -(k-self.m)) <= (k-self.m)/2 ):
                p += 1
            index = ns+self.m
            new_marks[j] = profile[j,index]
#        print("p = ",p/grad.shape[0])
        p = p/grad.shape[0]
        return new_marks,p

class ASM:
    
    def __init__(self,images,marks,scale,m): 
        self.PCAmodel = self.buildPCA(marks)
        
        imgs= rescale_img(images,scale)
        marks = rescale_marks(marks,scale)
        marks = marks.reshape((marks.shape[0],-1,80))
        self.glmodel = glm(imgs,marks,m)
        self.scale = scale
        self.m = m
        
    def buildPCA(self,marks):
        TableMarks,mean_shape = pa.GPA(marks)
        self.mean_shape = mean_shape        
        marks = np.asarray(TableMarks)
        accuracy = 0.98
        PCAmodel = PCA(marks,accuracy)
        return PCAmodel
        
    def init_start_by_rectangle(self,top_left,buttom_right):
        scale = 5.5*(buttom_right[0]-top_left[0])
        cent = np.zeros(2)
        cent[0] = (buttom_right[0]+top_left[0])/2
        cent[1] = (buttom_right[1]+top_left[1])/2
        init_mark = rescale_marks(self.mean_shape,scale)
        init_mark = pa.transform(init_mark,cent)
        init_mark = self.PCAmodel.addPoints(init_mark)
        self.init_mark = init_mark
        return init_mark
    def init_start_mark(self,mark):
        self.init_mark = mark
    def fit(self,test_image,p = 0.9,max_itr = 8):
        test_image = cv2.resize(test_image,(int(self.scale*test_image.shape[1]),int(self.scale*test_image.shape[0])))

        mark = self.init_mark
        
        mark = rescale_marks(mark,self.scale)
        picture = plotLandmarks(test_image,mark)
        if(picture.shape[0]>1000):
                picture = cv2.resize(picture,(1600,800))
        cv2.imshow('test',picture)
        cv2.waitKey(200)
        for i in range(max_itr):
            new_posi,p = self.glmodel.find_best(test_image,mark,k=self.m+2)
            mark = self.PCAmodel.addPoints(new_posi)
            picture = plotLandmarks(test_image,mark)
            if(picture.shape[0]>1000):
                picture = cv2.resize(picture,(1600,800))
            cv2.imshow('test',picture)
            k = cv2.waitKey(200) & 0xFF
            if k == 27:
                break
            if p >= 0.9:
                break
        cv2.destroyAllWindows()
        mark = rescale_marks(mark,1/self.scale)
        return mark
                    
class MultiResASM:
    def __init__(self,train_images,marks,number_of_levels = 4,m = 4):
        asms = []
        scale = 1/(2**number_of_levels)
        for i in range(number_of_levels+1):
            asms.append(ASM(train_images,marks,scale,m))
            scale = 2.0*scale
            
        self.asms = asms
    def fit(self,init,test_image):
#        print(init[0],init[1])
        mark = self.asms[0].init_start_by_rectangle(init[0],init[1])
        mark = self.asms[0].fit(test_image)
        for asm in self.asms[1:]:
            mark = asm.init_start_mark(mark)
            mark = asm.fit(test_image)
        return mark
