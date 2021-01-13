# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:22:03 2019

@author: steve
"""
import numpy as np
import math

class CFNN_network():
    def newCPN(P,T,delta,nseed,beta=0.5,alpha1=1):
    #P: the input arrays with npt*ndim
    #T: the output arrays with npt*ndim2
    #delta: Represents the distance to the center point 
    #beta: the output layer learning rate
    #alpha1: 1/alpha=the input layer learning rate
    #nseed: The initial number of rules
    #Wi: The rule's weight of the input layer
    #Wp: The rule's weight of the output layer
        [npt,ndim]=P.shape
        ndata=nseed
        seedIdx=np.random.permutation(npt)
        
        Wi=P[seedIdx[0],:]
        if len(T.shape)==1:
            T=np.reshape(T,(-1,1))
        Wp=T[seedIdx[0],:]
        Wi=np.asarray(Wi);Wp=np.asarray(Wp)
        numrule=1
        org_numrule=0
        while org_numrule!=numrule:
            org_numrule=numrule
            for i in range(0,ndata):
                if len(Wi.shape)==1:
                    dmin=np.sum((P[seedIdx[i],:]-Wi[:])**2)
                else:
                    dmin=np.sum((P[seedIdx[i],:]-Wi[0,:])**2)
                Idx=0
                for j in range(1,numrule):
                    dist=np.sum((P[seedIdx[i],:]-Wi[j,:])**2)
                    if dmin > dist:
                        dmin=dist
                        Idx=j
                if dmin<=delta:
                    if len(Wi.shape)==1:
                        Wi[:]=Wi[:]+1/alpha1*(P[seedIdx[i],:]-Wi[:])
                        Wp[:]=Wp[:]+beta*(T[seedIdx[i],:]-Wp[:])
                    else:
                        
                        Wi[Idx,:]=Wi[Idx,:]+1/alpha1*(P[seedIdx[i],:]-Wi[Idx,:])
                        Wp[Idx,:]=Wp[Idx,:]+beta*(T[seedIdx[i],:]-Wp[Idx,:])
                else:
                    if len(Wi.shape)==1:
                        Wi=np.reshape(Wi,(-1,len(Wi)))
                        Wp=np.reshape(Wp,(-1,len(Wp)))
                    Wi=np.append(Wi,[P[seedIdx[i],:]],axis=0)
                    Wp=np.append(Wp,[T[seedIdx[i],:]],axis=0)
        
                    numrule=numrule+1
            
            ndata=npt
            alpha1=alpha1+1
        return Wi,Wp
            
    def evalGaussCPN(Wi,Wp,delta,P):
        #Yh: the output value of the network
        #P: the input arrays with npt*ndim
        #Wi: The rule's weight of the input layer
        #Wp: The rule's weight of the output layer
        #delta: Half of the base length of the Gauss membership function
        
        if len(Wp.shape)==1:
            Wp=np.reshape(Wp,(-1,1))
        if len(Wi.shape)==1:
            Wi=np.reshape(Wi,(-1,1))
        numrule=Wi.shape[0]
        [npt,ndim]=P.shape
        D=np.zeros(numrule)
        s=np.zeros(numrule)
        Yh=np.zeros((npt,Wp.shape[1]))
        for i in range(0,npt):
            S=0
            delta2=delta*delta
            while S<0.00001:
                for j in range(0,numrule):
                    dist=np.sum((Wi[j,:]-P[i,:])**2)
                    D[j]=dist/delta2
                    s[j]=math.exp(-D[j])
                S=np.sum(s)
                delta2=4*delta2 #2 times larger than the original delta, the delta2 is the square of the delta. Thus, the coefficient is 4. 
            for k in range(0,Wp.shape[1]):    
                Yh[i,k]=np.sum(Wp[:,k]*s/S)
        return Yh
    
    def evalTriCPN(Wi,Wp,delta,P):
        #Yh: the output value of the network
        #P: the input arrays with npt*ndim
        #Wi: The rule's weight of the input layer
        #Wp: The rule's weight of the output layer
        #delta: Half of the base length of the traingle membership function
        
        if len(Wp.shape)==1:
            Wp=np.reshape(Wp,(-1,1))
        if len(Wi.shape)==1:
            Wi=np.reshape(Wi,(-1,1))
        numrule=Wi.shape[0]
        [npt,ndim]=P.shape
        D=np.zeros(numrule)
        s=np.zeros(numrule)
        Yh=np.zeros((npt,Wp.shape[1]))
        for i in range(0,npt):
            delta1=delta
            S=0
            while S<0.00001:
                for j in range(0,numrule):
                    dist=(np.sum((Wi[j,:]-P[i,:])**2))**0.5
                    if dist<delta1:
                        D[j]=dist/delta1
                    else:
                        D[j]=1
                    s[j]=1-D[j]
                S=np.sum(s)
                delta1=2*delta1
            for k in range(0,Wp.shape[1]):    
                Yh[i,k]=np.sum(Wp[:,k]*s/S)
        return Yh
    
    
    
