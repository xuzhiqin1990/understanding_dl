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
    
    
isuniform_l=0

#BaseDir = 'fitnd/'
subFolderName = '%s_isuni_%s'%(datetime.now().strftime("%y%m%d%H%M%S"),isuniform_l) 
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
nl_x = np.array([1000,2000,4000,8000,16000,50000])
#nl_x = np.array([500,1000])
nl_x = np.array([8000])
#nl_x=np.flip(nl_x,axis=0)
R['nl_x']=nl_x
    #hyper parameters
lossmethod = 'mse'
ctr_rbw = '111'
actfun = 'relu'
optmethod = 'gd' #or adam
#    optmethod = 'adam'
len_nlx=len(nl_x)

vl = 1
va = 0.2
vw = 0.8  #float('{:4g}'.format(I*10*vl/num_l1/va))

#lr = 6e-6
slr=[ 2e-5, 2e-5, 2e-5, 2e-6,2e-6,2e-6,2e-6]
#lr = 3e-5
epochs = int(3e6)

#number of testing samples
N = 30
mg = 1
b1_mg = 1

lossepoch=10

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

x_train = np.asarray([[1,1],[1,-1],[-1,1],[-1,-1]])
y_train = np.asarray([[1],[-1],[-1],[1]])

[xt1,xt2] = np.meshgrid(np.linspace(-1,1,N),np.linspace(-1,1,N))
x_tst = np.asarray([xt1.ravel(),xt2.ravel()]).transpose()  
R['x_train']=x_train
R['y_train']=y_train 
R['x_tst']=x_tst
for ii in range(len(nl_x)):
    print(ii)
    num_l1 = nl_x[ii]
    t0 = time.time()
    tf.reset_default_graph()
    lr=slr[ii]
    R['lr']=lr

    dimx = 2
    x = tf.placeholder(tf.float32, [None, dimx])
    merge_w = lambda x: np.concatenate([x,-x],axis = 0)
    merge_rb = lambda x: np.concatenate([x,x],axis = 1)
    
    ini_w1 = merge_rb(np.random.randn(dimx,int(num_l1/2)))*vw
    #ini_w1_r = np.sqrt(ini_w1[0:1,:]**2+ini_w1[1:2,:]**2)
#    ini_w1_r = np.ones([1,num_l1])*vr
#    ini_w1_e = ini_w1/ini_w1_r
    #ini_b = merge_rb((np.random.rand(1,int(num_l1/2))*2-1))*vb
#    ini_w = merge_w((np.random.rand(int(num_l1/2),1)*2-1))*vw
    ini_w2 = merge_w((np.random.randn(int(num_l1/2),1)))*va
    #overwrite initial value
    #ini_b = merge_rb((np.random.rand(1,int(num_l1/2))*2-1)*vl)*ini_w1
    if isuniform_l:
        ini_b = merge_rb((np.random.rand(1,int(num_l1/2))*2-1)*vl)
    else:
        ini_b = merge_rb((np.random.randn(1,int(num_l1/2))*2-1)*vl)
    
    #layer1
#    
    b1 = tf.get_variable('b1', [1, num_l1], initializer=tf.constant_initializer(ini_b),\
                         trainable = (ctr_rbw[1]=='1'))
    #layer2
    W2 = tf.get_variable('w2',[num_l1,1], initializer=tf.constant_initializer(ini_w2),\
                         trainable = (ctr_rbw[2]=='1'))
    
    
    W1 = tf.get_variable('w1', [dimx, num_l1], initializer=tf.constant_initializer(ini_w1),\
                         trainable = (ctr_rbw[0]=='1'))    
    y1 = tf.nn.relu(tf.matmul(x, W1)+b1)
    
    y2 = tf.matmul(y1, W2) 
    #output
    y = y2
    y_ = tf.placeholder(tf.float32, [None, 1])
    
    #################note:I use reduce_sum instead of reduce mean
    if lossmethod == 'mse': 
        mse = tf.reduce_sum(tf.reduce_sum((y_ -y)**2, reduction_indices=[1]))
    
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
    
    #x_tst = np.linspace(np.min(x_train)*mg,np.max(x_train)*mg,N+1)[:-1][:,np.newaxis]
    
    
    ma2=np.mean(abs(ini_w2)**2)
    mr3=np.mean(abs(ini_w1)**3)
    mr1=np.mean(abs(ini_w1))
    nf = np.max([mr3,ma2*mr1])
    y_tst2,y_trn2 = FI(np.squeeze(x_train),np.squeeze(y_train),np.squeeze(x_tst),mr3/nf,4*np.pi**2*b1_mg**2*ma2*mr1/nf)
    #y_tst2 = FI(x_train.ravel(),y_train.ravel(),x_tst.ravel(),mr3/nf,4*np.pi**2*b1_mg**2*ma2*mr1/nf,ep = 1e-6)
#    y_tst2 = FI(x_train.ravel(),y_train.ravel(),x_tst.ravel(),mr3/nf,b1_mg**2*ma2*mr1/nf,ep = 1e-6)
    R['y_tst2']=y_tst2
    fp=plt.figure()
    ax=plt.gca()
    ax.axis('equal')
    plt.title('reference, n=%s'%(R['nl_x'][ii]),fontsize=16)
    plt.plot(R['x_train'][:,0],R['x_train'][:,1],'w*',markersize=12)
    plt.scatter(R['x_tst'][:,0],R['x_tst'][:,1],marker = 's',c=R['y_tst2'],s=100 , cmap='rainbow')
    #cbar=plt.colorbar()
    #cbar.set_ticks([-1,0,1])
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    plt.clim(-1,1)
    plt.xlabel(r'x',fontsize=18)
    plt.ylabel('y',fontsize=18)
    #plt.cticks([-1,-0.5,0,0.5,1]) 
    ax.set_position(pos, which='both')
    fntmp = '%syFI_%s'%(FolderName,num_l1) 
    mySaveFig(plt,fntmp,ax=ax,isax=1,fp=fp,iseps=0)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True  
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    y_ini = sess.run(y,feed_dict = {x:x_tst})
    lossi = sess.run(mse, feed_dict={x: x_train, y_: y_train})
    
    loss_all = [] 
    
    for i in range(epochs):
        if lossi<1e-5:
            print('stop at epoch {}'.format(i))
            break
        
        _,lossi= sess.run([train_step,mse], feed_dict={x: x_train, y_: y_train})
        if (i%lossepoch==0):
            loss_all.append(lossi)
            print('nlx:%s, epoch: %s, loss: %s'%(nl_x[ii],i,lossi))
        if (i%1000==0) | (i==(epochs-1)):
            
            y_tst = sess.run(y, feed_dict={x: x_tst})

            v_err = err(np.squeeze(y_tst),y_tst2)
            fp=plt.figure()
            ax=plt.gca()
            plt.plot(loss_all,label='loss')
            plt.yscale('log')
            plt.xscale('log')
            plt.xlabel('x(*%s)'%(lossepoch),fontsize=18)
            fntmp = '%sloss'%(FolderName)
            mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
            
            R['y_tst']=y_tst
            fp=plt.figure()
            ax=plt.gca()
            ax.axis('equal')
            plt.title('DNN output, n=%s'%(R['nl_x'][ii]),fontsize=16)
            plt.plot(R['x_train'][:,0],R['x_train'][:,1],'w*',markersize=12)
            plt.scatter(R['x_tst'][:,0],R['x_tst'][:,1],marker = 's',c=np.squeeze(R['y_tst']),s=100 , cmap='rainbow')
            #cbar=plt.colorbar()
            #cbar.set_ticks([-1,0,1])
            ax.set_xticks([-1,0,1])
            ax.set_yticks([-1,0,1])
            plt.clim(-1,1)
            plt.xlabel(r'x',fontsize=18)
            plt.ylabel('y',fontsize=18)
            #plt.cticks([-1,-0.5,0,0.5,1])  
            ax.set_position(pos, which='both')
            fntmp = '%synn_%s'%(FolderName,num_l1) 
            mySaveFig(plt,fntmp,ax=ax,isax=1,fp=fp,iseps=0)
            
            fp=plt.figure()
            ax=plt.gca()
            plt.plot(R['y_tst'],R['y_tst2'], 'bo') 
            plt.plot(R['y_tst'],R['y_tst'], 'k-')
            plt.legend(fontsize=18)
            ax.set_xlabel('dnn',fontsize=18)
            ax.set_ylabel('predict',fontsize=18)
            plt.rc('xtick',labelsize=18)
            plt.rc('ytick',labelsize=18)
            ax.set_position(pos, which='both')
            plt.title('{}neuron'.format(R['nl_x'][ii]))
            fntmp = '%s%s%s'%(R['FolderName'],'compare',ii)
            mySaveFig(plt,fntmp,fp=fp,iseps=0,isShowPic=0)
           
    y_tst = sess.run(y, feed_dict={x: x_tst})

    v_err = err(np.squeeze(y_tst),y_tst2)
    
    all_var['errl1'].append(v_err[1])
    all_var['errl2'].append(v_err[0])
    
    R['loss_all']=loss_all
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
               
    R['y_tst']=y_tst
    fp=plt.figure()
    ax=plt.gca()
    ax.axis('equal')
    plt.title('DNN output, n=%s'%(R['nl_x'][ii]),fontsize=16)
    plt.plot(R['x_train'][:,0],R['x_train'][:,1],'w*',markersize=12)
    plt.scatter(R['x_tst'][:,0],R['x_tst'][:,1],marker = 's',c=np.squeeze(R['y_tst']),s=100 , cmap='rainbow')
    #cbar=plt.colorbar()
    #cbar.set_ticks([-1,0,1])
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    plt.clim(-1,1)
    plt.xlabel(r'x',fontsize=18)
    plt.ylabel('y',fontsize=18)
    #plt.cticks([-1,-0.5,0,0.5,1])  
    ax.set_position(pos, which='both')
    fntmp = '%synn_%s'%(FolderName,num_l1) 
    mySaveFig(plt,fntmp,ax=ax,isax=1,fp=fp,iseps=0)
     
    fp=plt.figure()
    ax=plt.gca()
    plt.plot(R['y_tst'],R['y_tst2'], 'r.',markersize=16) 
    plt.plot(R['y_tst'],R['y_tst'], 'k-',linewidth=4) 
    ftsz=28
    ax.set_xlabel(r'$f_{NN}$',fontsize=ftsz)
    ax.set_ylabel(r'$f_{LFP}$',fontsize=ftsz)
    plt.rc('xtick',labelsize=ftsz)
    plt.rc('ytick',labelsize=ftsz)
    #plt.axis('off')
    ax.set_position(pos, which='both')
    #plt.title('{}neuron'.format(R['nl_x'][ii]))
    fntmp = '%s%s%s'%(R['FolderName'],'compare',ii)
    mySaveFig(plt,fntmp,fp=fp,iseps=0,isShowPic=0)
    
    fp=plt.figure()
    ax=plt.gca()
    #plt.title('DNN output, n=%s'%(R['nl_x'][ii]),fontsize=16)
    plt.plot(R['x_train'][:,0],R['x_train'][:,1],'k*',markersize=16)
    plt.scatter(R['x_tst'][:,0],R['x_tst'][:,1],marker = 's',c=R['y_tst'][:,0],s=100 , cmap='rainbow')
    cbar=plt.colorbar()
    cbar.set_ticks([-1,0,1])
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    plt.clim(-1,1)
    ax.set_xlabel(r'$x_{1}$',fontsize=ftsz)
    ax.set_ylabel(r'$x_{2}$',fontsize=ftsz)
    plt.rc('xtick',labelsize=ftsz)
    plt.rc('ytick',labelsize=ftsz)
    plt.axis('on')
    fntmp = '%s%s%s'%(R['FolderName'],'DNNoutput',R['nl_x'][ii])
    mySaveFig(plt,fntmp,fp=fp,iseps=0,isShowPic=0)
    
#    plt.savefig(name.format('fig3'))
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
    plt.loglog(nl_x[0:ii+1],np.sqrt(l2),'*r-',nl_x[0:ii+1],l1,'*b-')
    plt.legend(['L2','L1'],fontsize=16)
    plt.xlabel('neuron num',fontsize=18)
    plt.ylabel('error',fontsize=18)
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    plt.title('error vs. neuron num',fontsize=18)
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









