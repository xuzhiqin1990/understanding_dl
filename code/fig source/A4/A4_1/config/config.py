# _*_ coding:utf-8 _*_

import argparse
import os

import yaml

# load config file(type:yaml)
config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='%s'%(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+os.sep+'config'+os.sep+'config.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
# general config
parser.add_argument('--device_rank', default='0',
                    help='path to dataset')
parser.add_argument('--t', type=float, default=1,
                    help='parameter initialization distribution variance power(We first assume that each layer is the same width.)')
parser.add_argument('--training_batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--test_batch_size', type=int, default=32,
                    help='input batch size for test (default: 32)')
parser.add_argument('--lr',  type=float, default=1e-3,
                    help='learning rate for training (default: 1e-3)')
parser.add_argument('--ini_output_dir', type=str,
                    default='/home/zhangzhongwang/data/saddle_points/test104/')
parser.add_argument('--input_dim',   default=1, type=int,
                    help='the input dimension for model (default: 1)')
parser.add_argument('--output_dim',   default=1, type=int,
                    help='the output dimension for model (default: 1)')
parser.add_argument('--training_size',   default=1000, type=int,
                    help='the training size for model (default: 1000)')
parser.add_argument('--test_size',   default=10000, type=int,
                    help='the test size for model (default: 10000)')
parser.add_argument('--plot_epoch',   default=1000, type=int,
                    help='step size of plotting interval (default: 1000)')
parser.add_argument('--save_epoch',   default=1000, type=int,
                    help='step size of saving interval (default: 1000)')
parser.add_argument('--training_steps',   default=100001, type=int,
                    help='the number of training steps (default: 100001)')
parser.add_argument('--stop_loss',   default=1e-5, type=float,
                    help='When the loss is less than the stop loss, the training ends. (default: 1e-5)')
parser.add_argument('--data',   default='1Dpro', type=str,
                    help='datasets type (default: 1Dpro)')
parser.add_argument('--hidden_layers_width',
                    nargs='+', type=int, default=[100])
parser.add_argument('--dropout',   default=False, type=bool,
                    help='need dropout or not(default:False)')
parser.add_argument('--dropout_pro',   default=0.2, type=float,
                    help='dropout proportion(default:0.2)')

parser.add_argument('--bias',   default=True, type=bool,
                    help='need bias or not')

parser.add_argument('--use_nesterov',   default=False, type=bool,
                    help='use nesterov optimizer or not')

parser.add_argument('--loss',   default='mse', type=str,
                    help='Loss function')

parser.add_argument('--save_para',   default=False, type=bool or int,
                    help='Whether to save the network parameters of each step of training. You can enter a boolean value or a list. If you enter a list, it means to save the value of the interval divided by the first element and the second element of the list.')

parser.add_argument('--save_grad',   default=False, type=bool or int,
                    help='Whether to save the network gradients of each step of training. You can enter a boolean value or a list. If you enter a list, it means to save the value of the interval divided by the first element and the second element of the list.')

parser.add_argument('--save_loss',   default=False, type=bool,
                    help='Whether to save the network loss of each step of training. ')


parser.add_argument('--save_acc',   default=False, type=bool,
                    help='Whether to save the network accuracy of each step of training. ')

parser.add_argument('--save_output',   default=False, type=bool or int,
                    help='wWether to save the network outputs of each step of training. You can enter a boolean value or a list. If you enter a list, it means to save the value of the interval divided by the first element and the second element of the list.')

parser.add_argument('--save_result_epoch',   default=1000, type=int,
                    help='step size of saving result dictionary (default: 1000)')

parser.add_argument('--resume_model', default='', help='resume model from checkpoint')

parser.add_argument('--target_func', default='',
                    help='target function')

parser.add_argument('--act_func', default='', help='activation function')


# 1Dpro config
parser.add_argument('--data_boundary', nargs='+', type=str, default=['-1', '1'],
                    help='the boundary of 1D data')
parser.add_argument('--Sampling times', default=1000, type=int,
                    help='times for sampling ')
parser.add_argument('--plot_output', default=True, type=bool,
                    help='times for dropout')
parser.add_argument('--load_index', default='0', type=str,
                    help='index of load model')


def parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()

    if args_config.config:
        with open(args_config.config, 'rb') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
