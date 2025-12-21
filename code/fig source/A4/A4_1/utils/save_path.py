import copy
import datetime
import os
import re
import shutil

import numpy as np
import torch
from torch.nn import parameter


def mkdir(fn):  # Create a directory
    if not os.path.isdir(fn):
        os.mkdir(fn)


def mkdirs(fn):  # Create directorys
    if not os.path.isdir(fn):
        os.makedirs(fn)


def create_save_dir(path_ori, m, t, load_path=''):
    """
    It creates a directory with a random name, and then creates a subdirectory called "output" inside of
    it
    
    :param path_ori: the path to the folder where the model will be saved
    :param m: model width
    :param t: initialization variance
    :param load_path: the path to the model you want to load
    :return: The path to the directory where the model will be saved.
    """

    mkdirs(path_ori)
    # path1 = f'{path_ori}{m}/'
    path1=os.path.join(path_ori, m)
    mkdir(path1)
    # path = f'{path1}{t}/'
    path=os.path.join(path1, str(t))
    mkdir(path)
    # subFolderName = '%s' % (
    # int(np.absolute(np.random.normal([1]) * 100000)) // int(1))
    subFolderName = re.sub(r'[^0-9]', '', str(datetime.datetime.now()))
    # subFolderName = int(load_path.split('.')[-3].split('p')[-1])
    # path = f'{path}{subFolderName}/'
    path = os.path.join(path, subFolderName)
    mkdir(path)

    # mkdir(f'{path}output/')

    mkdir(os.path.join(path, 'output'))

    return path


# It saves the model and optimizer state_dicts, and the args object, to a file.
class CheckpointSaver:
    def __init__(
            self,
            model,
            optimizer,
            args=None,
            path='',
            extension='.pth.tar'

    ):

        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.args = args

        # state
        # (filename, metric) tuples in order of decreasing betterness
        self.checkpoint_files = []
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''

        # config
        self.checkpoint_dir = path
        self.extension = extension

    def save_checkpoint(self, epoch):
        assert epoch >= 0
        self.args.model_dirs = os.path.join(self.checkpoint_dir, 'model')
        mkdir(self.args.model_dirs)
        tmp_save_path = os.path.join(
            self.args.model_dirs, f'tmp{self.extension}')
        self._save(tmp_save_path, epoch)

    def _save(self, save_path, epoch):
        save_state = {
            'epoch': epoch,
        }
        if self.model is not None:
            save_state['arch'] = type(self.model).__name__.lower(),
            save_state['state_dict'] = self.model.state_dict(),
        if self.optimizer is not None:
            save_state['optimizer'] = self.optimizer.defaults
        if self.args is not None:
            save_state['args'] = self.args
        torch.save(save_state, save_path)
        return save_state


# It saves the results of a training run
class ResultSaver:
    def __init__(
        self,
        args=None,
        path='',
        extension='.pth.tar'
    ):

        self.args = args
        self.checkpoint_dir = path
        self.extension = extension
        self.result_dict = {}
        self.save_keys = []

    def save_checkpoint(self, epoch):
        assert epoch >= 0
        tmp_save_path = os.path.join(
            self.checkpoint_dir, f'result{self.extension}')
        self._save(tmp_save_path)

    def create_result_dict(self):

        if self.args.save_para:
            self.result_dict['para'] = []
            self.save_keys.append('para')
            if isinstance(self.args.save_para, list):
                self.save_para_min, self.save_para_max = self.args.save_para[
                    0], self.args.save_para[1]

                self.save_para_interval=1
                
            elif isinstance(self.args.save_para, int):
                self.save_para_min, self.save_para_max = 0, np.inf
                self.save_para_interval = self.args.save_para
            else:
                self.save_para_min, self.save_para_max = 0, np.inf

                self.save_para_interval = 1

        if self.args.save_grad:
            self.result_dict['grad'] = []
            self.save_keys.append('grad')

            if isinstance(self.args.save_grad, list):
                self.save_grad_min, self.save_grad_max = self.args.save_grad[
                    0], self.args.save_grad[1]

                self.save_grad_interval=1

            elif isinstance(self.args.save_grad, int):
                self.save_grad_min, self.save_grad_max = 0, np.inf
                self.save_grad_interval = self.args.save_grad
            else:
                self.save_grad_min, self.save_grad_max = 0, np.inf

                self.save_grad_interval = 1

        if self.args.save_loss:
            self.save_keys.append('loss_train')
            self.save_keys.append('loss_test')
        self.result_dict['loss_train'] = []
        self.result_dict['loss_test'] = []
        if self.args.save_acc:
            self.result_dict['acc_train'] = []
            self.result_dict['acc_test'] = []
            self.save_keys.append('acc_train')
            self.save_keys.append('acc_test')
        if self.args.save_output or self.args.plot_output:
            self.result_dict['test_outputs'] = []
        if self.args.save_output:
            self.save_keys.append('test_outputs')

        return self.result_dict

    def _save(self, save_path):

        if self.save_keys != []:

            save_result_dict = dict(
                (k, self.result_dict[k]) for k in self.save_keys)

            torch.save(save_result_dict, save_path)

    def add_result(self, model, loss, loss_val, acc, acc_val, epoch):

        if self.args.save_grad and epoch >= self.save_grad_min and epoch < self.save_grad_max and epoch % self.save_grad_interval==0:
            grad_direc = []
            for name, p in model.named_parameters():
                grad_d = copy.deepcopy(p.grad.data.detach().cpu())
                grad_direc.append(grad_d)
            self.result_dict['grad'].append(grad_direc)

        if self.args.save_para and epoch >= self.save_para_min and epoch < self.save_para_max and epoch % self.save_para_interval == 0:

            self.result_dict['para'].append(copy.deepcopy(model.state_dict()))

        # if self.args.save_loss:
        self.result_dict['loss_train'].append(loss)
        self.result_dict['loss_test'].append(loss_val)

        if self.args.save_acc:
            self.result_dict['acc_train'].append(acc)
            self.result_dict['acc_test'].append(acc_val)

    def add_ini_result(self, model):

        self.result_dict['para'].append(copy.deepcopy(model.state_dict()))


def save_code_and_config(target_path):

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    shutil.copytree(path, os.path.join(target_path, 'code'))
