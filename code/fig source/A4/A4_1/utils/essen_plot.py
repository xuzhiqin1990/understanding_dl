import gc
import os
import warnings

# import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch.nn.functional as F
import yaml
from matplotlib import cm, ticker
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils.save_path import mkdir

warnings.filterwarnings("ignore")
matplotlib.use('Agg')


def save_fig(pltm, fntmp, pdf=True, x_log=True, y_log=True):
    """
    The function takes in a matplotlib object, a filename, and two boolean values. If the boolean values
    are true, the x and y axes are set to log scale. The function then saves the figure as a png and pdf
    file

    :param pltm: the plot object
    :param fntmp: The name of the file to save the figure to
    :param pdf: if True, save the figure as a pdf file, defaults to True (optional)
    :param x_log: If True, the x-axis will be logarithmic, defaults to True (optional)
    :param y_log: If True, the y-axis will be logarithmic, defaults to True (optional)
    """

    if x_log:
        plt.xscale('log')

    if y_log:
        plt.yscale('log')

    pltm.tight_layout()
    pltm.show()
    pltm.savefig('%s.png' % (fntmp))
    if pdf:
        pltm.savefig("%s.pdf" % (fntmp))


def plot_loss(path, loss_train, epoch_range=None, x_log=False):
    """
    It plots the loss of the training data

    :param path: the path to save the figure
    :param R: a dictionary containing the results of the training
    :param epoch_range: the range of epochs to plot
    :param x_log: whether to plot the x-axis in log scale, defaults to False (optional)
    """

    plt.figure()
    ax = plt.gca()
    y2 = np.asarray(loss_train)
    print(y2.shape)
    plt.plot(y2, 'k-', label='Train')
    if not epoch_range == None:
        plt.xlim(epoch_range[0], epoch_range[1])
    plt.title('loss', fontsize=15)
    if x_log == False:
        # fntmp = r'%s\loss' % (path)

        fntmp=os.path.join(path, 'loss')
        if epoch_range != None:
            fntmp += '_%s_%s' % (epoch_range[0], epoch_range[1])
        save_fig(plt, fntmp, x_log=False, y_log=True)
    else:
        fntmp = os.path.join(path, 'loss_log')
        # fntmp = r'%s\loss_xlog' % (path)
        if epoch_range != None:
            fntmp += '_%s_%s' % (epoch_range[0], epoch_range[1])
        save_fig(plt, fntmp, x_log=True, y_log=True)


def plot_model_output(path, args, argsy, epoch):
    """
    It plots the true values of the training data and the predicted values of the test data.

    :param path: the path to the directory where the model is saved
    :param args: a dictionary containing all the parameters for the training process
    :param argsy: a dictionary of the model outputs
    :param epoch: the current epoch number
    """
    if args.input_dim == 1:

        plt.figure()
        ax = plt.gca()

        plt.plot(args.train_inputs.detach().cpu().numpy(),
                 args.train_targets.detach().cpu().numpy(), 'b*', label='True')
        plt.plot(args.test_inputs.detach().cpu().numpy(),
                 argsy['test_outputs'][epoch].detach().cpu().numpy(), 'r-', label='Test')
        plt.title('output epoch=%s' % (epoch), fontsize=15)
        plt.legend(fontsize=18)
        fntmp = os.path.join(path, 'output', str(epoch))
        # fntmp = '%soutput/%s' % (path, epoch)
        save_fig(plt, fntmp, pdf=False, x_log=False, y_log=False)


def plot_loss_with_eig(path, loss, eig, eig_interval=1, lr=0.1):
    """
    It plots the loss function (in log scale) and the eigenvalues (in linear scale) on the same plot

    :param path: the path to the directory where the results are stored
    :param R: a dictionary containing the loss values for each epoch
    :param eig: the eigenvalues of the Hessian
    """

    plt.figure()
    ax = plt.gca()

    ax.plot(np.arange(1, len(loss)+1, eig_interval), eig, color='r', zorder=2)

    ax.set_ylabel('eigvalue')
    plt.axhline(y=2/lr, ls='--')
    ax2 = ax.twinx()

    ax2.plot(np.arange(1, len(loss)+1), loss, zorder=1)
    ax2.set_yscale('log')

    ax2.set_ylabel('loss')

    plt.xlabel('epoch')

    fntmp = r'%s\loss_eig_%s' % (path, len(loss))
    save_fig(plt, fntmp, x_log=False, y_log=True)

    fntmp = r'%s\loss_eig_log_%s' % (path, len(loss))
    save_fig(plt, fntmp, x_log=True, y_log=True)


def plot_contour_trajectory(X, Y, loss_all, x_lst, y_lst, path):
    """
    It plots a contour plot of the loss function, and then plots the trajectory of the optimizer on top
    of it

    :param X: the x-axis values of the contour plot
    :param Y: the y-axis values of the contour plot
    :param loss_all: the loss function evaluated at each point in the grid
    :param x_lst: list of x values
    :param y_lst: list of y values
    :param path: the path to save the figure
    """

    fig = plt.figure()
    CS1 = plt.contour(X, Y, loss_all)

    plt.clabel(CS1, inline=1, fontsize=8)
    plt.plot(x_lst, y_lst, '-o')

    fntmp = path + r'\_2dcontour_proj_all_2'

    save_fig(plt, fntmp, x_log=False, y_log=False)


def plot_contour(X, Y, loss_all, path):
    """
    It plots a contour plot of the loss function, and then plots the trajectory of the optimizer on top
    of it

    :param X: the x-axis values of the contour plot
    :param Y: the y-axis values of the contour plot
    :param loss_all: the loss function evaluated at each point in the grid
    :param x_lst: list of x values
    :param y_lst: list of y values
    :param path: the path to save the figure
    """

    fig = plt.figure()
    CS1 = plt.contour(X, Y, loss_all,locator=ticker.LogLocator(base=2))

    plt.clabel(CS1, inline=1, fontsize=8)

    fntmp=os.path.join(path, '2dcontour')


    # fntmp = path + r'\_2dcontour_proj_all_2'

    save_fig(plt, fntmp, x_log=False, y_log=False)


def plot_neuron(path, output_tensor, args):
    """
    It plots the true values of the training data and the predicted values of the test data of single neuron.

    :param path: the path to the directory where the model is saved
    :param output_tensor: Tensor output by each neuron
    :param args: a dictionary containing all the parameters for the training process
    """
    if args.input_dim == 1:

        output_tensor = output_tensor.squeeze().detach().cpu().numpy()

        mkdir(r'%s/neuron_output' % (path))

        for i in range(output_tensor.shape[1]):

            plt.figure()
            ax = plt.gca()

            plt.plot(args.train_inputs.detach().cpu().numpy(),
                     args.train_targets.detach().cpu().numpy(), 'b*', label='True')
            plt.plot(args.test_inputs.detach().cpu().numpy(),
                     output_tensor[:, i], 'r-', label='Test')
            plt.title('neuron index=%s' % (i), fontsize=15)
            plt.legend(fontsize=18)
            fntmp = r'%s/neuron_output/%s' % (path, i)
            save_fig(plt, fntmp, pdf=False, x_log=False, y_log=False)

    else:
        raise Exception('Input dimension must equal to 1!')


def plot_feature(path, ori, A, args, nota=''):
    """
    It plots the feature of the input data
    
    :param path: the path to save the figure
    :param ori: the feature orientation
    :param A: the feature amplitude
    :param args: the parameters of the model
    """

    if args.input_dim == 1:

        plt.figure()
        ax = plt.gca()

        if len(ori) == 1:

            ori = ori[0].squeeze().detach().cpu().numpy()
            A = A[0].squeeze().detach().cpu().numpy()

        elif len(ori) == 2:
            ori_ini, ori = ori[0].squeeze().detach().cpu(
            ).numpy(), ori[1].squeeze().detach().cpu().numpy()
            A_ini, A = A[0].squeeze().detach().cpu().numpy(
            ), A[1].squeeze().detach().cpu().numpy()

            print(ori_ini.shape, A_ini.shape)

            plt.scatter(ori_ini, abs(A_ini), color='cyan', label='ini feature')

        else:
            raise Exception('The length of the checkpoint list is less than or equal to two.')

        mkdir(r'%s/feature' % (path))

        plt.scatter(ori, abs(A), color='r', label='fin feature')
        plt.xlim(-3.16,3.16)
        plt.xlabel('orientation', fontsize=18)
        plt.ylabel('amplitude', fontsize=18)
        plt.legend(fontsize=18)
        fntmp = r'%s/feature/%s' % (path, nota)
        save_fig(plt, fntmp, pdf=False, x_log=False, y_log=True)

    else:
        raise Exception('Input dimension must equal to 1!')

def plot_loss_landscape(path, alpha, loss_all, index=''):
    """
    It plots the loss landscape for a given model
    
    :param path: the path to save the figure
    :param alpha: the interpolation parameter
    :param loss_all: the loss values for each alpha value
    :param index: the index of the interpolation
    """
    plt.figure()
    ax = plt.gca()
    plt.plot(alpha, loss_all, label='%s' % (index))
    plt.xlabel(r'$\alpha$', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.axvline(0.0, linestyle='--')
    plt.axvline(1.0, linestyle='--')
    plt.legend(fontsize=18)
    fntmp = r'%s/loss_interpolation' % (path)
    save_fig(plt, fntmp, pdf=False, x_log=False, y_log=True)


def plot_fft(y_fft,y_fft_pred, idx, path):
    """
    It plots the real and predicted FFTs, and saves the plot to a file
    
    :param y_fft: the real fft
    :param y_fft_pred: the predicted fft
    :param idx: the index of the frequency that we want to predict
    :param path: the path to the directory where the plots will be saved
    """
    plt.plot(y_fft+1e-5, label='real')
    plt.plot(idx, y_fft[idx]+1e-5, 'o')
    plt.plot(y_fft_pred+1e-5, label='train')
    plt.plot(idx, y_fft_pred[idx]+1e-5, 'o')
    plt.legend()
    plt.xlabel('freq idx')
    plt.ylabel('freq')
    fntmp = r'%s/fft' % (path)
    save_fig(plt, fntmp, pdf=False, x_log=False, y_log=True)


def plot_fft_hotmap(abs_err, path):
    """
    It plots a heatmap of the absolute error of the FFT of a signal
    
    :param abs_err: the absolute error of the FFT
    :param path: the path to the directory where the plots will be saved
    """

    plt.figure()
    plt.pcolor(abs_err, cmap='RdBu', vmin=0.1, vmax=1)
    plt.colorbar()
    fntmp = r'%s/fft_hotmap' % (path)
    save_fig(plt, fntmp, pdf=False, x_log=False, y_log=False)



def plot_eig_vs_var(path, var, eig, epoch, y_log=True, x_log=True):
    plt.figure()
    ax = plt.gca()
    plt.scatter(abs(eig), abs(np.array(var)))
    if y_log:
        ax.set_yscale('log')
    ax.set_ylim((1e-30, 10))
    if x_log:
        ax.set_xscale('log')
    ax.set_xlim((1e-5, 10))
    plt.xlabel('eigenvalue for hessian')
    plt.ylabel('variance')
    plt.title('eigenvalue v.s. variance', fontsize=15)
    plt.legend(fontsize=18)
    fntmp = '%seig_vs_varlog%s' % (path, epoch)
    save_fig(plt, fntmp, ax=ax, isax=1, iseps=0)

    plt.figure()
    ax = plt.gca()
    plt.scatter(abs(eig), abs(np.array(var)))
    eig_log = np.log10(abs(eig))
    var_log = np.log10(var)
    index = np.argsort(eig_log)[::-1][:4]
    coe = np.polyfit(eig_log[index], var_log[index], 1)
    plt.xlabel('eigenvalue for hessian')
    plt.ylabel('variance')
    plt.title('eigenvalue v.s. variance %.3f' % (coe[0]), fontsize=15)
    plt.legend(fontsize=18)
    fntmp = '%seig_vs_var%s' % (path, epoch)
    save_fig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_eig_vs_mean(path, mean, eig, epoch, y_log=True, x_log=True):
    plt.figure()
    ax = plt.gca()
    plt.scatter(abs(eig), mean)
    if y_log:
        ax.set_yscale('log')
    ax.set_ylim((1e-30, 10))
    if x_log:
        ax.set_xscale('log')
    ax.set_xlim((1e-5, 10))
    plt.xlabel('eigenvalue for hessian')
    plt.ylabel('mean')
    plt.title('eigenvalue v.s. mean', fontsize=15)
    plt.legend(fontsize=18)
    fntmp = '%seig_vs_meanlog%s' % (path, epoch)
    save_fig(plt, fntmp, ax=ax, isax=1, iseps=0)

    plt.figure()
    ax = plt.gca()
    plt.scatter(eig, mean)
    plt.xlabel('eigenvalue for hessian')
    plt.ylabel('mean')
    plt.title('eigenvalue v.s. mean', fontsize=15)
    plt.legend(fontsize=18)
    fntmp = '%seig_vs_mean%s' % (path, epoch)
    save_fig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_ori_A_trajectory(path, m, k, ori, A):
    fp = plt.figure()
    ax1 = plt.subplot(111, projection='polar')
    ax1.set_ylim(0, 1.1)
    for i in range(m):
        # if i % 10==0:
        #     print(i)
        line = ax1.plot(ori[i, :k+1], A[i, :k+1]**(0.1),
                        '-', lw=0.5, color='cyan', zorder=1)

        # ax1.scatter(ori[i,-1], A[i,-1],s=10)
    sca = ax1.scatter(ori[:, k], A[:, k]**(0.1), color='r', zorder=2, s=10)
    plt.savefig(
        '/home/zhangzhongwang/data/saddle_points/test96_retrain/2.0/200/101237/pic/%s.png' % (k))
    fp.clf()
    plt.close()
    gc.collect()


def plot_loss_one(path, loss, k, x_log=False):
    plt.figure()
    ax = plt.gca()
    y2 = np.asarray(loss)
    plt.plot(y2, 'k-', label='Train')
    plt.plot(k, loss[k], 'bo')
    if x_log:
        ax.set_xscale('log')
    ax.set_yscale('log')
    # plt.legend(fontsize=18)
    plt.title('loss', fontsize=15)
    if x_log == False:
        fntmp = '%s%s.png' % (path, k)
    else:
        fntmp = '%s%s.png' % (path, k)
    plt.savefig(fntmp)
    plt.clf()
    plt.close()
    gc.collect()


def concen_pic(save_path, image_column, image_row, path1, path2, i, weigh=640, height=480):
    to_image = Image.new('RGB', (image_column * weigh,
                         image_row * height))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    # for y in range(1, IMAGE_ROW + 1):
    #     for x in range(1, IMAGE_COLUMN + 1):
    from_image_1 = Image.open('%s%s.png' % (path1, i))
    from_image_2 = Image.open('%s%s.png' % (path2, i))
    # from_image_3 = Image.open('%s%s.png' % (path2, i))
    # from_image_4 = Image.open('%s%s.png' % (path4, i))
    to_image.paste(from_image_1, (0, 0))
    to_image.paste(from_image_2, (0, height))
    # to_image.paste(from_image_3, (weigh, 0))
    # to_image.paste(from_image_4, (weigh, height))
    return to_image.save("%s%s.png" % (save_path, i))  # 保存新图


def images_to_video(save_path, video_folder, rep=5, result_filename=None):

    if result_filename is None:
        result_filename = "{}.avi".format(save_path)
    images_name = {int(os.path.splitext(f)[0]): os.path.join(
        video_folder, f) for f in os.listdir(video_folder)}
    img = cv2.imread(images_name[0])
    height, width, layers = img.shape
    print(height)
    print(width)
    four_cc = cv2.VideoWriter_fourcc(*"XVID")  # avi
    video = cv2.VideoWriter(result_filename, four_cc, 25, (width, height))
    print(len(images_name))
    for i in range(len(images_name)):
        for j in range(rep):
            img = cv2.imread(images_name[int(5*i)])
            video.write(img)
        if i % 100 == 0:
            print("Done {}%".format((i*100)/len(images_name)))
    cv2.destroyAllWindows()
    video.release()
    print("Done!")
    return None


def plot_sigma_F(path):
    lst = os.listdir(path)
    plt.figure()
    ax = plt.gca()
    for i in lst:
        if i.startswith('p'):
            continue
        yamlPath = '%s%s/code/config/config.yaml' % (path, i)
        with open(yamlPath, 'r', encoding='utf-8') as f:
            config = f.read()
        d = yaml.load(config, Loader=yaml.FullLoader)
        sigma = np.loadtxt('%s%s/sigma.txt' % (path, i))
        theta_nage = np.loadtxt('%s%s/theta_nage.txt' % (path, i))
        theta_posi = np.loadtxt('%s%s/theta_posi.txt' % (path, i))
        if int(d['training_size']) > int(d['training_batch_size']):
            kind = 'SGD'
            plt.scatter(10**(theta_nage)+10**(theta_posi), abs(sigma/10000),
                        label='batch size:%s, lr:%s' % (d['training_batch_size'], d['lr']))
        elif d['dropout']:
            kind = 'dropout'
            plt.scatter(10**(theta_nage)+10**(theta_posi), abs(sigma/10000),
                        label='p:%s, lr:%s' % (1-float(d['dropout_pro']), d['lr']))
        elif d['add_tru_on_weight']:
            kind = 'add tru on weight'
            plt.scatter(10**(theta_nage)+10**(theta_posi), abs(sigma/10000),
                        label='turblence:%s, lr:%s' % (d['turblence'], d['lr']))
        elif d['add_tru_on_grad']:
            kind = 'add tru on grad'
            plt.scatter(10**(theta_nage)+10**(theta_posi), abs(sigma/10000),
                        label='turblence:%s, lr:%s' % (d['turblence'], d['lr']))
        else:
            kind = 'GD'
            plt.scatter(10**(theta_nage)+10**(theta_posi), abs(sigma/10000),
                        label='training size:%s, lr:%s' % (d['training_size'], d['lr']))
    # plt.plot([10**(0.3),10**(-1)],[10**(-9),10**(-6)],'--')
    # plt.plot([10**(-0.5),10],[10**(-6.1),10**(-8)],'--')
    # plt.title(kind+', bias:%s, no training after the selected point:%s'%(d['bias'], d['pca_with_no_training']))
    plt.xlabel(r'$F_{v_{i}(\Sigma)}$', fontsize=18)
    plt.ylabel(r'$\lambda_{i}(\Sigma)$', fontsize=18)
    my_leg = plt.legend(fontsize=18, loc='upper right')
    # plt.legend()
    # ax.set_xlim((1e-1, 1e1))
    # ax.set_ylim((1e-9, 2*10**(-5)))
    # ax.set_xlim((1e-1, 1e1))
    # ax.set_ylim((3e-9, 7e-7))
    ax.tick_params(labelsize=18)
    # ax.text(2,10**(-6.9),r'$slope = -1.3$', fontsize=18)
    # ax.text(10**(-0.9),1e-8,r'$slope = -2.3$', fontsize=18)
    # x_vals = np.array(ax.get_xlim())
    # print(x_vals)
    # y_vals = np.array(ax.get_ylim())
    # print(y_vals)
    # y_vals_1 = 10**(-14+ -4* np.log10(x_vals) )
    # print(y_vals_1)
    # plt.plot([10**(-0.9),10**(0.1)],[1e-6,1e-10] , '--')
    # plt.xlim(x_vals)
    # plt.ylim(y_vals)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('%spic.png' % (path))
    plt.savefig('%spic.pdf' % (path))



# def plot_several_loss_landscape(path,alpha, loss_all):
#     plt.figure()
#     ax = plt.gca()
#     for ind,loss_lst in enumerate (loss_all):
#         plt.plot(alpha+ind,loss_lst)
#     plt.yscale('log')
#     plt.savefig('%sloss_landscape.png'%(path))


def plot_several_loss_landscape(path, alpha, loss_all):
    plt.figure()
    ax = plt.gca()
    label = ['before shock', 'after shock']
    for ind, loss_lst in enumerate(loss_all):
        plt.plot(alpha, loss_lst, label=label[ind])

    # plt.yscale('log')
    plt.xlabel(r'$\alpha$', fontsize=18)
    plt.ylabel(r'loss', fontsize=18)
    ax.tick_params(labelsize=18)

    # plt.ylim((0,60))
    # plt.xticks([-1.6,-0.8,0,0.8,1.6])
    # plt.yticks([0,20,40,60])
    plt.tight_layout()
    plt.legend(fontsize=18)
    plt.savefig('%sloss_landscape_random.png' % (path))
    # plt.savefig('%sloss_landscape_random.pdf'%(path))


def plot_cov_hessian(path):
    lst = os.listdir(path)

    plt.figure()
    ax = plt.gca()
    color = ['b', 'g', 'c', 'm', 'y']
    for ind, i in enumerate(lst):
        if i.startswith('p'):
            continue
        yamlPath = '%s%s/code/config/config.yaml' % (path, i)
        with open(yamlPath, 'r', encoding='utf-8') as f:
            config = f.read()
        d = yaml.load(config, Loader=yaml.FullLoader)
        iden_trace = np.loadtxt('%s%s/iden_trace.txt' % (path, i))
        ini_trace = np.loadtxt('%s%s/ini_trace.txt' % (path, i))
        # index=np.arange(0,len(ini_trace))*5
        plt.plot(ini_trace[:30], '-', color=color[ind],
                 label='p: %s, lr: %s' % (1-float(d['dropout_pro']), d['lr']))
        plt.plot(iden_trace[:30], '--', color=color[ind])
    plt.xticks([0, 5, 10, 15, 20, 25, 30], [
               0, 50, 100, 150, 200, 250, 300], fontsize=18)

    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('trace', fontsize=18)
    # plt.ylim((1e-7,1e-1))
    # ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim((1e-7, 1e-1))
    ax.tick_params(labelsize=18)
    ax.set_yticks([1e-1, 1e-3, 1e-5, 1e-7])
    # plt.yticks([1e-1,1e-3,1e-5,1e-7],fontsize=18)
    # plt.yticks([1e-7,1e-5,1e-3,1e-1],fontsize=18)
    plt.legend(loc='best', fontsize=18)
    plt.yscale('log')
    ax.text(5, 1e-3, r'$Tr(H \Sigma)$', fontsize=18)
    ax.text(5, 1e-5, r'$Tr(H \bar{\Sigma})$', fontsize=18)
    plt.tight_layout()
    plt.savefig('%spic.pdf' % (path))
