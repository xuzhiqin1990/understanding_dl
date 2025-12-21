# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:18:55 2019

@author: agate
"""

import numpy as np
import matplotlib.pyplot as plt

def Fbase_1d(x,rg = 2,f_num = 200,res_refine = 10):
    f = 2*np.pi*(np.arange(f_num*res_refine))*(1/(rg*res_refine))
    f_ = np.asarray([f,f]).transpose().ravel()
    base = []
    for k in f:
        base.append(np.cos(2*np.pi*k*x))
        base.append(np.sin(2*np.pi*k*x))
    ET = np.asarray(base)
    return ET.transpose(),ET,f_       

def Fbase_2d(x,rg = 2,f_num = 10,res_refine = 4):
#    [X1,X2]=np.meshgrid(x,x) 
#    X1 = X1.ravel()
#    X2 = X2.ravel()
    X1 = x[:,0]
    X2 = x[:,1]
    
    f = 2*np.pi*(np.arange(f_num*res_refine))*(1/(rg*res_refine))
    f_ = []
    base = []
    for k1 in f:
        for k2 in f:
            base.append(np.cos(k1*X1+k2*X2))
            base.append(np.sin(k1*X1+k2*X2))
            base.append(np.cos(k1*X1-k2*X2))
            base.append(np.sin(k1*X1-k2*X2))
            f_.append(np.sqrt(k1**2+k2**2))
    f_ = np.asarray([f_,f_,f_,f_]).transpose().ravel()
    ET = np.asarray(base)
    return ET.transpose(),ET,f_ 

def Fbase(x):
    if np.ndim(x) == 1:
        ETt,ET,f_  = Fbase_1d(x)
    if np.ndim(x) == 2:
        ETt,ET,f_  = Fbase_2d(x)
    return ETt,ET,f_


#def Fbase(x):
#    if np.ndim(x) == 1:
#        ETt,ET,f_  = Fbase_1d(x)
#    if np.ndim(x) == 2:
#        ETt,ET,f_  = Fbase_2d(x)
#    return ETt,ET,f_
    
def csch(x):
    tmp=np.exp(x)
    return 2/(tmp-1/tmp)

def get_tanh_wk(f__,sa,sr):
    wf=[]
    for tmp_f in f__:
        cc=csch(np.pi**2*tmp_f/sr)**2/sr*1e10
#        if np.min(cc)<1e-6:
#            print('cc<1e-6 f:%s  cc:%s'%(tmp_f,cc))
        tmp2=4 * np.pi**2 * sa**2 / sr**2 * tmp_f**2
        tmp3=cc*(1+tmp2)
        wf.append(np.mean(tmp3))
    return np.asarray(wf)
    
    
def FI(x_train,y_train,x_tst,rvar,wrvar,istanh=0, ep = 1e-6):
    E,ET,f_ = Fbase(x_train)
    print(np.shape(E))
    print(np.shape(ET))
    print(np.shape(f_))
    #NI = x_train[0].shape[0]
    NI = np.ndim(x_train) 
    
    print(NI)
    C = ET@E
    f__ = f_
    f__[:NI*2] = np.ones(NI*2)*f_[NI*2]*0.1
    #preparing weight matrix
    
    
#    NI = NI+1
    
    if istanh==0:
        C = ET@E
        Wk = rvar/f__**(NI+3)+wrvar/f__**(NI+1)
        Winv = np.diag(Wk**-1)
        #    print(np.max(Wk.ravel()))
        #solving step
        Fk = np.linalg.solve(C+ep*Winv,ET@y_train)
    else:
        Wk=get_tanh_wk(f__,rvar,wrvar)
        ind=np.where(Wk>ep)[0]
        tWk=Wk[ind]
        tWinv = np.diag(tWk**-1)
        tE=E[:,ind]
        tET=ET[ind,:]
        tC=tET@tE
        tFk = np.linalg.solve(tC+ep*tWinv,tET@y_train)
        Fk=np.zeros([len(f__),1])
        Fk[ind,0]=tFk
        print('Wk min:')
        print(np.min(Wk))
        
    
    #reconstructing y
    E_full = Fbase(x_tst)[0]
    y_tst = E_full@Fk
    #print(np.shape(E))
    y_trn = E@Fk
    return y_tst,y_trn

def FI_ini(x_train,y_train,x_tst,y_init,rvar,wrvar,ep = 1e-6):
    E,ET,f_ = Fbase(x_train)
    C = ET@E
    f__ = f_
    f__[:NI*2] = np.ones(NI*2)*f_[NI*2]*0.1
    #preparing weight matrix
    NI = len(x_train[0])
    Wk = rvar/f__**(NI+3)+wrvar/f__**(NI+1)
    Winv = np.diag(Wk**-1)
    E_full = Fbase(x_tst)[0]
    C_full = E_full.transpose()@E_full
    a0 = np.linalg.solve(C_full+ep*Winv,E_full.transpose()@y_init)    
    #solving step
    Fk = np.linalg.solve(C+ep*Winv,ET@y_train+ep*Winv@a0)
    #reconstructing y   
    y_tst = E_full@Fk
    return y_tst

#rvar = 0.1
#wvar = 0.1
#wrvar = 0.01
#
#r0 = 0.64
#N = 50
#L = 10
#theta_cord = np.linspace(0,2*np.pi,200)
#train_X = np.zeros([2,len(theta_cord)])
#r = r0*(1+0.5*np.sin(L*theta_cord))
#train_X[0,:] = r*np.cos(theta_cord)
#train_X[1,:] = r*np.sin(theta_cord)
##train_Y = np.sign(np.sin(N*theta_cord+1e-9))
#train_Y = np.sin(N*theta_cord)
#x_train = train_X.transpose()
#y_train = train_Y[:,np.newaxis]
#
##x_train = np.asarray([[1,1],[1,-1],[-1,1],[-1,-1]])
##y_train = np.asarray([[1],[-1],[-1],[1]])
#[xt1,xt2] = np.meshgrid(np.linspace(-1,1,101),np.linspace(-1,1,101))
#x_tst = np.asarray([xt1.ravel(),xt2.ravel()]).transpose() 
#y_tst2,y_trn2 = FI(np.squeeze(x_train),np.squeeze(y_train),np.squeeze(x_tst),(rvar+wvar),wrvar,ep = 1e-7)
#
#import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
##fig = plt.figure()
##ax = plt.axes(projection='3d')
##ax.scatter3D(x_tst[:,0],x_tst[:,1],y_tst2)
##plt.show()
##
##fig = plt.figure()
##ax = plt.axes(projection='3d')
##ax.plot_surface(xt1,xt2,y_tst2.reshape(xt1.shape))
##plt.show()
#
#plt.figure()
#plt.subplot(2,2,1)
#plt.scatter(train_X[0,:],train_X[1,:],c=train_Y,s=100 , cmap='rainbow')
#plt.colorbar()
#plt.subplot(2,2,2)
#plt.scatter(x_tst[:,0],x_tst[:,1],c=y_tst2,s=100 , cmap='rainbow')
#plt.colorbar()
#plt.subplot(2,2,3)
#plt.scatter(x_tst[:,0],x_tst[:,1],c=y_tst2,s=100 , cmap='rainbow')
#plt.colorbar()
#plt.clim(-1,1)
#plt.subplot(2,2,4)
#plt.scatter(train_Y,y_trn2,s=20)
#plt.plot(train_Y,train_Y,'-k')
#plt.show()