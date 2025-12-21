import random

import numpy as np
import torch


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    """
    It computes the gradient of the outputs with respect to the inputs
    
    :param outputs: the output of the function you want to differentiate
    :param inputs: the input tensor
    :param grad_outputs: a Tensor or a list of tensors that will be used to compute the gradient. If it
    is a Tensor, then the gradient of the sum of outputs with respect to it will be returned. If it is a
    list, then the gradient of the sum of outputs with respect to each element
    :param retain_graph: If False, the graph used to compute the grad will be freed. Note that in nearly
    all cases setting this option to True is not needed and often can be worked around in a much more
    efficient way. Defaults to the value of create_graph
    :param create_graph: If true, graph of the derivative will be constructed, allowing to compute
    higher order derivative products. Defaults to False, defaults to False (optional)
    :return: The gradient of the output with respect to the input.
    """
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(
        y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])


def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    """
    It computes the Hessian of the output with respect to the inputs
    
    :param output: the output of the function you want to differentiate
    :param inputs: the input tensor
    :param out: the output of the function
    :param allow_unused: If True, then the function will return a Hessian matrix with rows and columns
    corresponding to only those input elements that were actually used in computing the output. If
    False, then the Hessian matrix will have rows and columns corresponding to all input elements,
    defaults to False (optional)
    :param create_graph: If True, graph of the derivative will be constructed, allowing to compute
    higher order derivative products. Defaults to False, defaults to False (optional)
    :return: The Hessian matrix of the output with respect to the inputs.
    """
    #     assert output.ndimension() == 0
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    print(n)
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        # print(i)
        [grad] = torch.autograd.grad(
            output, inp, create_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)
        for j in range(inp.numel()):
            # print(j)
            if grad[j].requires_grad:
                row = gradient(
                    grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel()
                                        for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        del grad

    return out

class derivatives:
    def __init__(
            self,
            model,
            loss_fn,
            args=None
    ):
        self.args = args
        self.model = model
        self.loss_fn = loss_fn

    def gradient(self, outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
        if torch.is_tensor(inputs):
            inputs = [inputs]
        else:
            inputs = list(inputs)
        grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                    allow_unused=True,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)
        grads = [x if x is not None else torch.zeros_like(
            y) for x, y in zip(grads, inputs)]
        return torch.cat([x.contiguous().view(-1) for x in grads])

    def hessian(self, output, inputs, out=None, allow_unused=False, create_graph=False):
        #     assert output.ndimension() == 0
        if torch.is_tensor(inputs):
            inputs = [inputs]
        else:
            inputs = list(inputs)

        n = sum(p.numel() for p in inputs)
        if out is None:
            out = output.new_zeros(n, n)

        ai = 0
        for i, inp in enumerate(inputs):
            [grad] = torch.autograd.grad(
                output, inp, create_graph=True, allow_unused=allow_unused)
            grad = torch.zeros_like(inp) if grad is None else grad
            grad = grad.contiguous().view(-1)
            for j in range(inp.numel()):
                if grad[j].requires_grad:
                    row = self.gradient(
                        grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
                else:
                    row = grad[j].new_zeros(sum(x.numel()
                                            for x in inputs[i:]) - j)

                out[ai, ai:].add_(row.type_as(out))  # ai's row
                if ai + 1 < n:
                    out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
                del row
                ai += 1
            del grad

        return out

    def first_order_derivatives(self, data_loader):
        for _, (data, target) in enumerate(data_loader, 1):
            data, target=data.to(self.args.device), target.to(self.args.device)
            outputs = self.model(data)
            loss = self.loss_fn(outputs, target)
            return self.gradient(loss, self.model.parameters())

    def second_order_derivatives(self, data_loader):
        for _, (data, target) in enumerate(data_loader, 1):
            data, target=data.to(self.args.device), target.to(self.args.device)
            outputs = self.model(data)
            loss = self.loss_fn(outputs, target)
            return self.hessian(loss, self.model.parameters())


class get_hessian_eig:
    def __init__(
            self,
            model,
            full_dataloader,
            loss_fn,
            dropout_size=None, 
            dropout_times=None, 
            args=None
    ):
        self.args = args
        self.model = model
        # self.train_loader_lst = train_loader_lst
        self.full_dataloader = full_dataloader
        self.loss_fn = loss_fn
        self.Derivatives = derivatives(self.model, self.loss_fn, self.args)
        self.dropout_times=dropout_times
        self.dropout_size=dropout_size

    def calcu_batch_deri(self):
        first_derivative_lst = []
        for data_loader in self.train_loader_lst:
            first_derivative = self.Derivatives.first_order_derivatives(
                data_loader)
            first_derivative_lst.append(
                first_derivative.detach().cpu().numpy())
        return first_derivative_lst

    def calcu_full_hessain(self):
        return self.Derivatives.second_order_derivatives(self.full_dataloader)

    def calcu_full_grad(self):
        return self.Derivatives.first_order_derivatives(self.full_dataloader)

    def calcu_cov(self):
        first_derivative_lst = self.calcu_batch_deri()
        first_derivative_lst=np.array(first_derivative_lst)
        cov=np.cov(first_derivative_lst.transpose())
        print(cov.shape)
        return cov
        # print(first_derivative_lst.shape)
    def calcu_hessian_eig(self):
        hessian = self.calcu_full_hessain()
        w, v = np.linalg.eigh(hessian.detach().cpu().numpy())
        return np.real(w), np.real(v)

    def calcu_projection(self,first_derivative_lst=None):
        w, v = self.calcu_hessian_eig()
        # first_derivative_lst = self.calcu_batch_deri()
        # if first_derivative_lst.all()==None:
        #     first_derivative_lst = self.calcu_dropout_lst()
        print(first_derivative_lst)
        U = np.zeros((len(w), len(first_derivative_lst)))
        for j in range(len(w)):
            for i in range(len(first_derivative_lst)):
                U[j, i] = np.dot(v[:, j], first_derivative_lst[i])
        return U, w
    
    def calcu_dropout_lst(self):
        first_order_derivatives=self.Derivatives.first_order_derivatives(self.full_dataloader)
        print(first_order_derivatives)
        first_order_derivatives_lst=[]
        for _ in range(self.dropout_times):
            indices = list(range(first_order_derivatives.detach().cpu().numpy().shape[0]))
            random.shuffle(indices)
            train_indices = indices[self.dropout_size:]
            index_matrix=np.ones_like(first_order_derivatives)
            index_matrix[train_indices]=0
            first_order_derivatives_lst.append(first_order_derivatives*index_matrix)
        return first_order_derivatives_lst
    

    def calcu_cov_eig(self):
        cov=self.calcu_cov()
        w, v = np.linalg.eig(cov)        
        return w,v
    
    def get_eig_and_proj(self):
        w,v=self.calcu_cov_eig()
        full_grad=self.calcu_full_grad()
        proj=[]
        for i in range(w.shape[0]):
            proj.append(np.dot(full_grad,v[:,i]))
        return proj,w


        




    def get_eigvalue_and_var(self,first_derivative_lst=None):
        U, w = self.calcu_projection(first_derivative_lst)
        return np.var(U, axis=1), w

    # def calau_hessian_eig(self):
