import numpy as np
from sklearn.decomposition import PCA
from model.linear import Linear
import torch.nn as nn
import torch
from .derivatives_of_parameters import one_hot
import os

class Get_weight_matrix_and_pca:
    def __init__(self, R, args, loss_fn, train_loader, para_dict,index):
        self.R = R
        self.index=index
        self.weight_matrix = self.get_weight_matrix()  # [TM/B,2500]
        _, self.pca_matrix = self.get_pca_matrix()# [2500weights,2500n_components]
        self.args = args
        self.loss_fn = loss_fn
        self.train_loader=train_loader
        self.para_dict=para_dict
      

    def get_weight_matrix(self):
        return np.array(self.R['exploration_para'])

    def get_pca_matrix(self):
        X = self.weight_matrix
        pca = PCA()
        pca.fit(X)
        pca_vector=[]
        print(self.index)
        for i in self.index:
            pca_vector.append(pca.components_[i])
        # centered_matrix = X - X.mean(axis=0)
        # cov = np.dot(centered_matrix.T, centered_matrix) 
        # eigvals, eigvecs = np.linalg.eig(cov) 
        # print(eigvals)
        # pca = PCA(n_components=2500)
        # pca.fit(matrix)
        # print(pca.singular_values_)
        # matrix = pca.transform(matrix)
        # norm=np.linalg.norm(matrix,ord=2,axis=0)
        # print(matrix.shape)  # [2500weights,2500n_components]
        # print(np.linalg.norm(matrix/norm,ord=2,axis=0))
        
        return pca.singular_values_[self.index],pca_vector

    # def get_thera_matrix(self):

    #     theta_matrix = np.zeros_like(self.weight_matrix)
    #     # for i in range(theta_matrix.shape[0]):
    #     for ind,i in enumerate( self.index):
    #         for j in range(theta_matrix.shape[1]):
    #             theta_matrix[ind, j] = np.dot(
    #                 self.weight_matrix[:, j], self.pca_matrix[:, i])

    #     return theta_matrix

    # def get_sigma2(self):
    #     sigma2_vector = np.zeros(len(self.index))
    #     theta_matrix = self.get_thera_matrix()  # [20,TM/B]
    #     for i in range(len(self.index)):
    #         sigma2_vector[i] = sum((theta_matrix[i, :-1])
    #                                ** 2)/theta_matrix.shape[1]
    #     print('sigma finished')
    #     return sigma2_vector  # [20]

    def construct_weight_matrix(self, theta, i):  # i=0,...,2499
        initial_weight = self.weight_matrix[0, :]

        return initial_weight+theta*np.array(self.pca_matrix[i])

    def getloss(self, theta, i):
        weight_matrix = self.construct_weight_matrix(theta, i)
        loss = get_loss_for_weight_matrix(
            weight_matrix, self.para_dict, self.train_loader, loss_fn=self.loss_fn, args=self.args)
        return loss

    def get_theta(self,i,loss_ini,ini_a=-3,ini_b=3):
        
        theta_posi=self.binary_search_posi(ini_a,ini_b,loss_ini,i)
    
        theta_nage=self.binary_search_nage(ini_a,ini_b,loss_ini,i)
        return theta_posi,theta_nage

    def binary_search_posi(self,ini_a,ini_b, loss_ini, i):
        a=ini_a
        b=ini_b
        if (b-a) == 0:
            return False
        midnum = (b+a) / 2
        loss=self.getloss(10**midnum,i,)
        # log_train_file = os.path.join(self.args.path, 'posi_steps.log')
        # with open(log_train_file, 'a') as log_po:
        #     log_po.write('index:%s,value:%.8f, error: %.8f \n'%(i,midnum,loss - 2*loss_ini))
        
        
        print('value:%.8f, error: %.8f'%(midnum,loss - 2*loss_ini))
        if abs(loss - 2*loss_ini)<2e-6:
            return midnum
        return self.binary_search_posi(a,midnum,loss_ini, i) if loss - 2*loss_ini >2e-6 else self.binary_search_posi(midnum,b,loss_ini, i)
    
    def binary_search_nage(self,ini_a,ini_b, loss_ini, i):
        a=ini_a
        b=ini_b
        if (b-a) == 0:
            return False
        midnum = (b+a) / 2
        loss=self.getloss(-10**midnum,i,)
        # log_valid_file = os.path.join(self.args.path, 'nega_steps.log')
        # with open(log_valid_file, 'a') as log_po:
        #     log_po.write('index:%s, value:%.8f, error: %.8f \n'%(i,midnum,loss - 2*loss_ini))
        print('value:%.8f, error: %.8f'%(midnum,loss - 2*loss_ini))
        if abs(loss - 2*loss_ini)<2e-6:
            return midnum
        return self.binary_search_nage(a,midnum,loss_ini, i) if loss - 2*loss_ini >2e-6 else self.binary_search_nage(midnum,b,loss_ini, i)

    def get_landscape_fig(self,theta,i):
        loss_all=[]
        for j in theta:
            loss=self.getloss(j,i)
            loss_all.append(loss)

        return loss_all


class cal_flatness:
    def __init__(self, args, loss_fn, train_loader, para_dict,weight0):
        self.args = args
        self.loss_fn = loss_fn
        self.train_loader=train_loader
        self.para_dict=para_dict
        self.weight0=weight0

    def construct_weight_matrix(self, theta, direction):  # i=0,...,2499
        initial_weight = self.weight0

        return initial_weight+theta*np.array(direction)

    def getloss(self, theta, direction):
        weight_matrix = self.construct_weight_matrix(theta, direction)
        loss = get_loss_for_weight_matrix(
            weight_matrix, self.para_dict, self.train_loader, loss_fn=self.loss_fn, args=self.args)
        return loss

    def get_theta(self,direction,loss_ini,ini_a=-3,ini_b=3):
        
        theta_posi=self.binary_search_posi(ini_a,ini_b,loss_ini,direction)
    
        theta_nage=self.binary_search_nage(ini_a,ini_b,loss_ini,direction)
        return theta_posi,theta_nage

    def binary_search_posi(self,ini_a,ini_b, loss_ini, direction):
        a=ini_a
        b=ini_b
        if (b-a) == 0:
            return False
        midnum = (b+a) / 2
        loss=self.getloss(10**midnum,direction)
        # log_train_file = os.path.join(self.args.path, 'posi_steps.log')
        # with open(log_train_file, 'a') as log_po:
        #     log_po.write('index:%s,value:%.8f, error: %.8f \n'%(i,midnum,loss - 2*loss_ini))
        
        
        print('value:%.8f, error: %.8f'%(midnum,loss - 2*loss_ini))
        if abs(loss - 2*loss_ini)<2e-6:
            return midnum
        return self.binary_search_posi(a,midnum,loss_ini, direction) if loss - 2*loss_ini >2e-6 else self.binary_search_posi(midnum,b,loss_ini, direction)
    
    def binary_search_nage(self,ini_a,ini_b, loss_ini, direction):
        a=ini_a
        b=ini_b
        if (b-a) == 0:
            return False
        midnum = (b+a) / 2
        loss=self.getloss(-10**midnum,direction)
        # log_valid_file = os.path.join(self.args.path, 'nega_steps.log')
        # with open(log_valid_file, 'a') as log_po:
        #     log_po.write('index:%s, value:%.8f, error: %.8f \n'%(i,midnum,loss - 2*loss_ini))
        print('value:%.8f, error: %.8f'%(midnum,loss - 2*loss_ini))
        if abs(loss - 2*loss_ini)<2e-6:
            return midnum
        return self.binary_search_nage(a,midnum,loss_ini, direction) if loss - 2*loss_ini >2e-6 else self.binary_search_nage(midnum,b,loss_ini, direction)


def get_loss_for_weight_matrix(weight_matrix, para_dict, train_loader, loss_fn=nn.CrossEntropyLoss(), args=None):
    device = args.device
    runing_loss = 0.0
    model = Linear(args.t, args.hidden_layers_width, args.input_dim,
                   args.output_dim, nn.ReLU(), args.initialization,args.dropout,args.dropout_pro,args.bias).to(args.device)

    
    para_dict['features.2.weight'] = torch.from_numpy(
        weight_matrix[:2500].reshape((50, -1))).to(args.device)

    if args.bias:
        para_dict['features.2.bias'] = torch.from_numpy(
        weight_matrix[2500:]).to(args.device)

    model.load_state_dict(para_dict)
    model.eval()

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        inputs = data
        outputs = model(inputs)
        if args.softmax:
            outputs = torch.nn.functional.softmax(outputs)
        if args.one_hot:
            target_onehot = one_hot(target, args.output_dim).to(device)
            loss = loss_fn(outputs, target_onehot.long())
        else:
            loss = loss_fn(outputs, target.long())

        runing_loss += loss.item()
    return runing_loss


# def get_batch_weight_matrix(train_loader_lst):
#     for data_loader in train_loader_lst:





