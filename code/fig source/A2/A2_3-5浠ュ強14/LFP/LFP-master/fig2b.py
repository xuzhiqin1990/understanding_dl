# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:16:15 2019

@author: agate
"""
#import matplotlib
#matplotlib.use('Agg')

from curvefitting_FI import *
from datetime import datetime
import os,sys
import matplotlib
matplotlib.use('Agg')   
import pickle
import time  
import shutil 
import platform
sys.path.insert(0,'../basicfolder') 
from BasicFunc import mySaveFig,mkdir
Leftp=0.18
Bottomp=0.18
Widthp=0.88-Leftp
Heightp=0.9-Bottomp
pos=[Leftp,Bottomp,Widthp,Heightp]

import matplotlib.pyplot as plt

import numpy as np  
import tensorflow as tf
tf.reset_default_graph()

from scipy.interpolate import CubicSpline

Leftp=0.18
Bottomp=0.18
Widthp=0.88-Leftp
Heightp=0.9-Bottomp
pos=[Leftp,Bottomp,Widthp,Heightp]


R={}  ### used for saved all parameters and data
 
sBaseDir = 'lfp/'

lenarg=np.shape(sys.argv)[0] #Sys.argv[ ]其实就是一个列表，里边的项为用户输入的参数，关键就是要明白这参数是从程序外部输入的，而非代码本身的什么地方，要想看到它的效果就应该将程序保存了，从外部来运行程序并给出参数。
if lenarg>1:
    ilen=1
    while ilen<lenarg: 
        if sys.argv[ilen]=='-g':
            R['gpu']=[np.int32(sys.argv[ilen+1])] 
        if sys.argv[ilen]=='-lr':
            R['learning_rate']=np.float32(sys.argv[ilen+1])  
        if sys.argv[ilen]=='-seed':
            R['seed']=np.int32(sys.argv[ilen+1])
        if sys.argv[ilen]=='-step':
            R['Total_Step']=np.int32(sys.argv[ilen+1])
        if sys.argv[ilen]=='-inputd':
            R['input_dim']=np.int32(sys.argv[ilen+1])
        if sys.argv[ilen]=='-tol':
            R['tol']=np.float32(sys.argv[ilen+1])
        if sys.argv[ilen]=='-act':
            R['ActFuc']=np.int32(sys.argv[ilen+1])
        ilen=ilen+2
        
        
print(R) 

R['gpu']='0'
os.environ["CUDA_VISIBLE_DEVICES"]='%s'%(R['gpu']) 


### mkdir a folder to save all output

if platform.system()=='Windows':
    device_n="1"
    BaseDir = '../../../nn/%s'%(sBaseDir)
else:
    device_n="3"
    BaseDir = sBaseDir
    matplotlib.use('Agg')
    
#BaseDir = 'fitnd/'
subFolderName = '%s'%(datetime.now().strftime("%y%m%d%H%M%S")) 
#subFolderName = '%s'%(int(np.absolute(np.random.normal([1])*100000))//int(1)) 
FolderName = '%s%s/'%(BaseDir,subFolderName)
mkdir(BaseDir)
mkdir(FolderName)
#if R['issave']:
#    mkdir('%smodel/'%(FolderName))
R['FolderName']=FolderName 

if  True: #not platform.system()=='Windows':
    shutil.copy(__file__,'%s%s'%(FolderName,os.path.basename(__file__)))
    

R['FolderName']=FolderName   ### folder for save images

#plt.figure()
#plt.plot(R['test_inputs'],R['y_true_test'])

t0=time.time() 


def err(y_tst,y_tst2):
    return [np.mean(np.mean((y_tst2.ravel()-y_tst.ravel())**2)),np.mean(np.mean(np.abs(y_tst2.ravel()-y_tst.ravel())))]

results = []
all_var={}
all_var['errl1']=[]
all_var['errl2']=[]
all_var['Rall']=[]
nl_x = np.array([2000,4000,8000,20000])
#nl_x = np.array([20000])
nl_x = np.array([10000])
nl_x=np.flip(nl_x,axis=0)
R['nl_x']=nl_x
    #hyper parameters
lossmethod = 'mse'
ctr_rbw = '111'
actfun = 'relu'
optmethod = 'gd' #or adam
#    optmethod = 'adam'
len_nlx=len(nl_x)
R['cscase']=1

R['istanh']=1
if R['cscase']==1:
    vl = 3
    va = 0.1
    vw = 3  #float('{:4g}'.format(I*10*vl/num_l1/va))
    slr=[ 1e-5,1e-5,1e-5,5e-6]
    slr=np.flip(slr,axis=0)
else:
    vl = 1
    va = 1
    vw = 0.2
    slr=[ 1e-6,1e-5,1e-5,1e-5]
if R['istanh']==1:
    if R['cscase']==1:
        vl = 3
        va = 4
        vw = 1.5  #float('{:4g}'.format(I*10*vl/num_l1/va))
        slr=[ 2e-5]
        slr=np.flip(slr,axis=0)
    else:
        vl = 1
        va = 0.1
        vw = 3
        slr=[ 1e-6,1e-5,1e-5,1e-5]
#lr = 6e-6
#slr=[ 6e-5, 6e-5, 6e-5, 6e-5,2e-6,2e-6,2e-6]
#lr = 3e-5
epochs = int(3e6)

#number of testing samples
N = 800
mg = 1
b1_mg = 1

lossepoch=2500

R['lossmethod']=lossmethod
R['ctr_rbw']=ctr_rbw
R['actfun']=actfun
R['optmethod']=optmethod

R['vl']=vl
R['va']=va
R['vw']=vw
R['epochs']=epochs
R['N']=N
R['mg']=mg
R['b1_mg']=b1_mg

R['lossepoch']=lossepoch

#def pwl(x,x0,y0):
#    y = np.zeros(x.shape)
#    for i in range(len(x0)-1):
#        idx = (x>=x0[i]) & (x<x0[i+1])
##        print(np.sum(idx))
#        y[idx] = ((x[idx]-x0[i])*y0[i+1]+(x0[i+1]-x[idx])*y0[i])/(x0[i+1]-x0[i])
#    return y
#
#x_tgt = np.array([[-1],[-0.7],[-0.1],[0.1],[1]])
#y_tgt = np.array([[1],[-1],[-0.3],[0.3],[1]])
##x_train = np.array([-1,-0.9,-0.8,-0.65,-0.4,-0.2,-0.06,0.05,0.2,0.4,0.7])[:-1,np.newaxis]
#x_train = np.array([-0.8,-0.65,-0.4,-0.2,-0.06,0.05,0.2])[:-1,np.newaxis]
#y_train = pwl(x_train,x_tgt,y_tgt)

x_train = np.array([-0.8,-0.6,-0.3,-0.06,0.1])[:,np.newaxis]
y_train = np.array([-0.5+0.75,-1+0.75,-0.82+0.75,-0.6+0.75,-0.8+0.75])[:,np.newaxis]     
x_tst = np.linspace(np.min(x_train)*mg,np.max(x_train)*mg,N+1)[:-1][:,np.newaxis]
if R['cscase']==1:
    cs = CubicSpline(np.squeeze(x_train), np.squeeze(y_train))
    R['ideal']=cs(np.squeeze(x_tst))
    plt.figure()
    plt.plot(x_train,y_train,'*-')
    plt.plot(np.squeeze(x_tst),cs(np.squeeze(x_tst)),'--')
for ii in range(len(nl_x)):
    print(ii)
    num_l1 = nl_x[ii]
    t0 = time.time()
    tf.reset_default_graph()
    lr=slr[ii]  #[ii]
    R['lr']=lr

    x = tf.placeholder(tf.float32, [None, 1])
    merge_a = lambda x: np.concatenate([x,-x],axis = 0)
    merge_wb = lambda x: np.concatenate([x,x],axis = 1)
    
    ini_w = merge_wb((np.random.rand(1,int(num_l1/2))*2-1))*vw
    #ini_b = merge_wb((np.random.rand(1,int(num_l1/2))*2-1))*vb
    ini_a = merge_a((np.random.rand(int(num_l1/2),1)*2-1))*va
    #overwrite initial value
    #ini_b = merge_wb((np.random.rand(1,int(num_l1/2))*2-1)*vl)*ini_r
    ini_b = merge_wb((np.random.rand(1,int(num_l1/2))*2-1)*vl)
    
    
    
    #layer1
    W1 = tf.get_variable('w1', [1, num_l1], initializer=tf.constant_initializer(ini_w),\
                         trainable = (ctr_rbw[0]=='1'))  
    b1 = tf.get_variable('b1', [1, num_l1], initializer=tf.constant_initializer(ini_b),\
                         trainable = (ctr_rbw[1]=='1'))
    #layer2
    W2 = tf.get_variable('w2',[num_l1,1], initializer=tf.constant_initializer(ini_a),\
                         trainable = (ctr_rbw[2]=='1'))
    
    b2 = tf.get_variable('b2',[1,], initializer=tf.keras.initializers.Zeros())
    if R['istanh']==0:
        y1 = tf.nn.relu(tf.matmul(x, W1)+b1_mg*b1) 
    else: 
        y1 = tf.nn.tanh(tf.matmul(x, W1)+b1_mg*b1) 
    y2 = tf.matmul(y1, W2) #+ b2
    
    Bnorm = tf.reduce_sum(tf.abs(W1*tf.transpose(W2))) 
    #output
    y = y2
    y_ = tf.placeholder(tf.float32, [None, 1])
    
    #################note:I use reduce_sum instead of reduce mean
    if lossmethod == 'mse': 
        mse = tf.reduce_sum((y_ -y)**2)
    
    if optmethod == 'gd':
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(mse)
    elif optmethod == 'adam':
        train_step = tf.train.AdamOptimizer(lr).minimize(mse)
    
#    gd_b = tf.gradients(ys = mse,xs = b1)[0]
#    new_b = b1.assign(b1-10*lr*gd_b)
    
    
    
    
    
    
#    plt.figure()
#    plt.plot(x_train,y_train,'*-')
    #x_train = np.array([[-1],[-0.7],[-0.1],[0.1],[1]])
    #y_train = np.array([[1],[-1],[-0.2],[0.2],[1]])
    
    
    #x_tst = np.linspace(np.min(x_train),np.max(x_train),N+1)[:-1][:,np.newaxis]
    
    
    if R['istanh']==0:
        ma2=np.mean(abs(ini_a)**2)
        mr3=np.mean(abs(ini_w)**3)
        mr1=np.mean(abs(ini_w))
        nf = np.max([mr3,ma2*mr1])
        y_tst2 = FI(x_train.ravel(),y_train.ravel(),x_tst.ravel(),mr3/nf,4*np.pi**2*b1_mg**2*ma2*mr1/nf,ep = 1e-6)
    else:
        y_tst2 = FI(x_train.ravel(),y_train.ravel(),x_tst.ravel(),rvar=np.squeeze(ini_a),wrvar=np.squeeze(np.abs(ini_w)),istanh=R['istanh'],ep = 1e-6)
#    y_tst2 = FI(x_train.ravel(),y_train.ravel(),x_tst.ravel(),mr3/nf,b1_mg**2*ma2*mr1/nf,ep = 1e-6)
    fp=plt.figure()
    ax=plt.gca()
    plt.plot(x_train,y_train,'k*', markersize=12,label='samples')
    #plt.plot(x_tgt,y_tgt,'k-',linewidth=2,label='auxiliary') 
    plt.plot(x_tst,y_tst2[0],'r--',linewidth=2,label='prediction') 
    plt.xlabel('x',fontsize=18)
    plt.ylabel('y',fontsize=18)
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    plt.title('neuron num: %s'%(num_l1),fontsize=18)
    plt.legend(fontsize=16)
    ax.set_position(pos, which='both')
    fntmp = '%ssnrmdiff_%s'%(BaseDir,num_l1)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True  
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    y_ini = sess.run(y,feed_dict = {x:x_tst})
    lossi,ai,wi,bi,Bnormi = sess.run([mse,W2,W1,b1,Bnorm], feed_dict={x: x_train, y_: y_train})
    
    loss_all = [] 
    
    for i in range(epochs):
        if lossi<8e-5:
            print('stop at epoch {}'.format(i))
            break
        
        
        
        if (i%10000==0) | (i==(epochs-1)):
            
            y_tst = sess.run(y, feed_dict={x: x_tst})

            v_err = err(y_tst,y_tst2[0])
            fp=plt.figure()
            ax=plt.gca()
            plt.plot(loss_all,label='loss')
            plt.yscale('log')
            plt.xscale('log')
            plt.xlabel('x(*%s)'%(lossepoch),fontsize=18)
            fntmp = '%sloss'%(FolderName)
            mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
            
            fp=plt.figure()
            ax=plt.gca()
            if R['istanh']==0:
                if R['cscase']==1:
                    plt.plot(np.squeeze(x_tst),cs(np.squeeze(x_tst)),'--',color='grey',linewidth=4,label='cubic-spline')
                else:
                    plt.plot(np.squeeze(x_train),np.squeeze(y_train),'--',color='grey',linewidth=4,label='linear-spline')
            #plt.plot(x_tgt,y_tgt,'k-',linewidth=2,label='auxiliary')
            plt.plot(x_tst,y_tst,'r-',linewidth=4,label=r'$f_{NN}$')
            plt.plot(x_tst,y_tst2[0],'b-.',linewidth=4,label=r'$f_{LFP}$')
            plt.plot(x_train,y_train,'k*', markersize=14,label='samples')
            #plt.plot(x_tst,y_ini,'c-',linewidth=2,label='initial-nn')
            ftsz=28
            plt.xlabel('x',fontsize=ftsz)
            plt.ylabel('y',fontsize=ftsz,rotation=0)
            plt.rc('xtick',labelsize=ftsz)
            plt.rc('ytick',labelsize=ftsz)
            plt.xticks([-0.8,-0.4,0],fontsize=ftsz)
            plt.yticks(fontsize=ftsz) 
            #plt.legend(fontsize=18,ncol=2)
            #plt.ylim([-1.25,-0.15])
            ax.set_position(pos, which='both')
            fntmp = '%ssnrmdiff_%s'%(FolderName,num_l1) 
            mySaveFig(plt,fntmp,ax=ax,isax=1,fp=fp,iseps=0)
        _,lossi,ai,wi,bi,Bnormi = sess.run([train_step,mse,W2,W1,b1,Bnorm], feed_dict={x: x_train, y_: y_train})
        if (i%lossepoch==0):
            loss_all.append(lossi)
            print('nlx:%s, epoch: %s, loss: %s'%(nl_x[ii],i,lossi))
    y_tst = sess.run(y, feed_dict={x: x_tst})

    v_err = err(y_tst,y_tst2[0])
    
    all_var['errl1'].append(v_err[1])
    all_var['errl2'].append(v_err[0])
    
    R['loss_all']=loss_all
    R['x_train']=x_train
    R['y_train']=y_train 
    R['x_tst']=x_tst
    R['y_tst']=y_tst
    R['y_tst2']=y_tst2
    R['y_ini']=y_ini
    R['num_l1']=num_l1   
    R['v_err']=v_err         
    
    fp=plt.figure()
    ax=plt.gca()
    plt.plot(loss_all,label='loss')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('x(*%s)'%(lossepoch),fontsize=18)
    fntmp = '%sloss%s'%(FolderName,num_l1)
    mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
               
    fp=plt.figure()
    ax=plt.gca()
    if R['istanh']==0:
        if R['cscase']==1:
            plt.plot(np.squeeze(R['x_tst']),cs(np.squeeze(R['x_tst'])),'--',color='grey',linewidth=4,label='cubic-spline')
        else:
            plt.plot(np.squeeze(R['x_train']),np.squeeze(R['y_train']),'--',color='grey',linewidth=4,label='linear-spline')
    #plt.plot(x_tgt,y_tgt,'k-',linewidth=2,label='auxiliary')
    plt.plot(R['x_tst'],R['y_tst'],'r-',linewidth=4,label=r'$f_{NN}$')
    plt.plot(R['x_tst'],R['y_tst2'][0],'b-.',linewidth=4,label=r'$f_{LFP}$')
    plt.plot(R['x_train'],R['y_train'],'k*', markersize=14,label='samples')
    #plt.plot(x_tst,y_ini,'c-',linewidth=2,label='initial-nn')
    ftsz=28
    plt.xlabel('x',fontsize=ftsz)
    plt.ylabel('y',fontsize=ftsz)
    plt.rc('xtick',labelsize=ftsz)
    plt.rc('ytick',labelsize=ftsz)
    plt.xticks([-0.8,-0.4,0],fontsize=ftsz)
    plt.yticks(fontsize=ftsz) 
    #plt.legend(fontsize=18,ncol=2)
    #plt.ylim([-1.25,-0.15])
    ax.set_position(pos, which='both')
    fntmp = '%ssnrmdiff_%s'%(FolderName,num_l1) 
    mySaveFig(plt,fntmp,ax=ax,isax=1,fp=fp,iseps=0)
    
#    
    all_var['Rall'].append(R)
    print(time.time()-t0)
    with open('%sobjs.pkl'%(FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(all_var, f, protocol=4)
    with open('%sR%s.pkl'%(FolderName,num_l1), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(R, f, protocol=4)
    text_file = open("%sOutput.txt"%(FolderName), "w")
    for para in R:
        if np.size(R[para])>20:
            continue
        text_file.write('%s: %s\n'%(para,R[para]))
        
        
    fp=plt.figure()
    ax=plt.gca()
    l2 = all_var['errl2']
    l1 = all_var['errl1']
    #plt.loglog(nl_x[0:ii+1],np.sqrt(l2),'*r-',nl_x[0:ii+1],l1,'*b-')
    plt.loglog(nl_x[0:ii+1],np.sqrt(l2),'*b-')
    #plt.legend(['L2','L1'],fontsize=16)
    plt.xlabel('neuron num',fontsize=ftsz)
    plt.ylabel('error',fontsize=ftsz)
    plt.rc('xtick',labelsize=ftsz)
    plt.rc('ytick',labelsize=ftsz)
    #plt.title('error vs. neuron num',fontsize=18)
    plt.grid(which='both')
    ax.set_position(pos, which='both')
    fntmp = '%serrorvsnum'%(FolderName)
    mySaveFig(plt, fntmp,ax=ax,fp=fp,isShowPic=0)



with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    all_var  = pickle.load(f)

nl_x = all_var['Rall'][-1]['nl_x']
#nl_x = np.array([500,1000])
#nl_x = np.array([50000])
#nl_x=np.flip(nl_x,axis=0)
#R['nl_x']=nl_x

target_folder=os.getcwd()
all_sub=os.listdir(target_folder) 
FolderName='' 

for sub in all_sub:
    print(sub)
    if not sub[-4:] == '.pkl' or sub=='objs.pkl':
        continue
    fd=target_folder+'/'+sub
    with open(fd, 'rb') as f:  # Python 3: open(..., 'rb')
        R  = pickle.load(f)
    print(R['lr'])
    print(R['num_l1'])
    fp=plt.figure()
    ax=plt.gca()
    plt.plot(R['x_train'],R['y_train'],'k*', markersize=12,label='samples')
    #plt.plot(x_tgt,y_tgt,'k-',linewidth=2,label='auxiliary')
    plt.plot(R['x_tst'],R['y_tst'],'b-',linewidth=2,label='final-nn')
    plt.plot(R['x_tst'],R['y_tst2'][0],'r--',linewidth=2,label='prediction')
    plt.plot(R['x_tst'],R['y_ini'],'c-',linewidth=2,label='initial-nn')
    plt.xlabel('x',fontsize=18)
    plt.ylabel('y',fontsize=18)
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    plt.title('neuron num: %s'%(R['num_l1']),fontsize=18)
    plt.legend(fontsize=16)
    ax.set_position(pos, which='both')
    fntmp = '%ssnrmdiff_%s'%(FolderName,R['num_l1'])
    mySaveFig(plt, fntmp,ax=ax,fp=fp,isShowPic=0)
    

fp=plt.figure()
ax=plt.gca()
l2 = all_var['errl2']
l1 = all_var['errl1']
#plt.loglog(nl_x[0:ii+1],np.sqrt(l2),'*r-',nl_x[0:ii+1],l1,'*b-')
#plt.legend(['L2','L1'],fontsize=16)
plt.loglog(nl_x[0:-1],np.sqrt(l2[:-1]),'*r-',label='L2')
plt.legend(fontsize=16)
plt.xlabel('neuron num',fontsize=18)
plt.ylabel('error',fontsize=18)
plt.ylim([1e-3,1e-1])
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.title('error vs. neuron num',fontsize=18)
plt.grid(which='both')
ax.set_position(pos, which='both')
fntmp = '%serrorvsnum'%(FolderName)
mySaveFig(plt, fntmp,ax=ax,fp=fp,isShowPic=0)

#erw_all = np.asarray([x[2] for x in results])
#print(erw_all)
#print(np.mean(erw_all,axis = 0))
#
#yp_mean = np.mean(np.asarray([x[3] for x in results]),axis = 0)
#yr_mean = np.mean(np.asarray([x[4] for x in results]),axis = 0)
#ym_err = err(yp_mean,yr_mean)
#plt.figure()
#plt.plot(x_train,y_train,'k*')
#plt.plot(x_tgt,y_tgt,'k-')
#plt.plot(x_tst,yp_mean,'b-')
#plt.plot(x_tst,yr_mean,'r--')
#plt.title('ym_err:{:6g},1norm:{:6g}'.format(*ym_err))
#plt.show()
#print(ym_err)


#plt.figure()
#nl_x = np.array([500,1000,2000,4000,8000,16000])
#l2 = np.array([1.33e-4,5e-5, 3.47e-5,2e-5,1.32e-5,7.78e-6])
#l1 = np.array([6.6e-3,4.4e-3,3.7e-3,2.8e-3,2.4e-3,1.85e-3])
#plt.loglog(nl_x,np.sqrt(l2),'*r-',nl_x,l1,'*b-')
#plt.loglog(nl_x,np.sqrt(l2[0])/(nl_x/nl_x[0])**0.33/1.5,'k--')
#plt.legend(['L2','L1','ref -1/3'])
#plt.grid(which='both')
#plt.show()
##plt.savefig('figures_cf\PC_scan_multi_fixed\InNote\I40_error.png')
#fp=plt.figure()
#ax=plt.gca()
#nl_x = np.array([500,1000,2000,4000,8000,16000])
#l2 = np.array([1.33e-4,5e-5, 3.47e-5,2e-5,1.32e-5,7.78e-6])
#l1 = np.array([6.6e-3,4.4e-3,3.7e-3,2.8e-3,2.4e-3,1.85e-3])
#plt.loglog(nl_x,np.sqrt(l2),'*r-',nl_x,l1,'*b-')
#plt.loglog(nl_x,np.sqrt(l2[0])/(nl_x/nl_x[0])**0.33/1.5,'k--')
#plt.legend(['L2','L1','ref -1/3'],fontsize=16)
#plt.xlabel('neuron num',fontsize=18)
#plt.ylabel('error',fontsize=18)
#plt.rc('xtick',labelsize=18)
#plt.rc('ytick',labelsize=18)
#plt.title('error vs. neuron num',fontsize=18)
#plt.grid(which='both')     









