import copy
import math
import warnings
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from custom_losses import ContractiveLoss
from custom_mnist import FastMNIST, NoisyMNIST


# set device globally
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# these variables will be allocated only if needed
mnist_train = None
mnist_test = None
noisy_mnist_train = None
noisy_mnist_test = None


def get_clean_sets():
    global mnist_train
    global mnist_test
    if mnist_train is None:
        mnist_train = FastMNIST(root='../MNIST/', train=True, download=True, transform=transforms.ToTensor())
        mnist_test = FastMNIST(root='../MNIST/', train=False, download=True, transform=transforms.ToTensor())
    return mnist_train, mnist_test


def get_noisy_sets(**kwargs):
    global noisy_mnist_train
    global noisy_mnist_test
    if noisy_mnist_train is None:
        noisy_mnist_train = NoisyMNIST(root='../MNIST/', train=True, download=True, transform=transforms.ToTensor(), **kwargs)
        noisy_mnist_test = NoisyMNIST(root='../MNIST/', train=False, download=True, transform=transforms.ToTensor(), **kwargs)
    return noisy_mnist_train, noisy_mnist_test


def fit_ae(model, mode=None, tr_data=None, val_data=None, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
    """
    Training functions for the AEs
    :param model: model to train
    :param mode: (str) {'basic | 'contractive' | 'denoising'}
    :param tr_data: (optional) specific training data to use
    :param val_data: (optional) specific validation data to use
    :param num_epochs: (int) number of epochs
    :param bs: (int) batch size
    :param lr: (float) learning rate
    :param momentum: (float) momentum coefficient
    :return: history of training (like in Keras)
    """
    mode_values = (None, 'basic', 'contractive', 'denoising')
    assert 0 < lr < 1 and num_epochs > 0 and bs > 0 and 0 <= momentum < 1 and mode in mode_values

    # set the device: GPU if cuda is available, else CPU
    model.to(device)

    # set optimizer, loss type and datasets (depending on the type of AE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = ContractiveLoss(ae=model, lambd=1e-4) if mode == 'contractive' else nn.MSELoss()
    if mode == 'denoising':
        if tr_data is not None or val_data is not None:
            warnings.warn("'denoising' flag was set, so NoisyMNIST will be used for training and validation")
        noisy_train, noisy_val = get_noisy_sets(**kwargs)
        tr_data, tr_targets = noisy_train.data, noisy_train.targets
        val_data, val_targets = noisy_val.data, noisy_val.targets
        del noisy_train, noisy_val
    else:
        tr_set, val_set = get_clean_sets()
        if tr_data is None:
            tr_data, tr_targets = tr_set.data, tr_set.targets
        else:
            tr_data = tr_data.to(device)
            tr_targets = torch.flatten(copy.deepcopy(tr_data), start_dim=1)
        if val_data is None:
            val_data, val_targets = val_set.data, val_set.targets
        else:
            val_data = val_data.to(device)
            val_targets = torch.flatten(copy.deepcopy(val_data), start_dim=1)
        del tr_set, val_set
    if 'ConvAutoencoder' in model.__class__.__name__:
        val_bs = bs
        tr_data, tr_targets = tr_data.cpu(), tr_targets.cpu()
        val_data, val_targets = val_data.cpu(), val_targets.cpu()
    else:
        val_bs = None
    torch.cuda.empty_cache()

    # training cycle
    loss = None  # just to avoid reference before assigment
    history = {'tr_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        # training
        model.train()
        tr_loss = 0
        n_batches = math.ceil(len(tr_data) / bs)
        # shuffle
        indexes = torch.randperm(tr_data.shape[0])
        tr_data = tr_data[indexes]
        tr_targets = tr_targets[indexes]
        progbar = tqdm(range(n_batches), total=n_batches)
        progbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_idx in range(n_batches):
            # zero the gradient
            optimizer.zero_grad()
            # select a (mini)batch from the training set and compute net's outputs
            train_data_batch = tr_data[batch_idx * bs: batch_idx * bs + bs].to(device)
            train_targets_batch = tr_targets[batch_idx * bs: batch_idx * bs + bs].to(device)
            outputs = model(train_data_batch)
            # compute loss (flatten output in case of ConvAE. targets already flat)
            loss = criterion(torch.flatten(outputs, 1), train_targets_batch)
            tr_loss += loss.item()
            # propagate back the loss
            loss.backward()
            optimizer.step()
            # update progress bar
            progbar.update()
            progbar.set_postfix(train_loss=f"{loss.item():.4f}")
        last_batch_loss = loss.item()
        tr_loss /= n_batches
        history['tr_loss'].append(round(tr_loss, 5))

        # validation
        val_loss = evaluate(model=model, data=val_data, targets=val_targets, criterion=criterion, bs=val_bs)
        history['val_loss'].append(round(val_loss, 5))
        torch.cuda.empty_cache()
        progbar.set_postfix(train_loss=f"{last_batch_loss:.4f}", val_loss=f"{val_loss:.4f}")
        progbar.close()

        # simple early stopping mechanism
        if epoch >= 10:
            last_values = history['val_loss'][-10:]
            if (abs(last_values[-10] - last_values[-1]) <= 2e-5) or (last_values[-3] < last_values[-2] < last_values[-1]):
                return history

    return history


def evaluate(model, criterion, mode='basic', data=None, targets=None, bs=None, **kwargs):
    """ Evaluate the model """
    # set the data
    if data is None:
        _, val_set = get_noisy_sets(**kwargs) if mode == 'denoising' else get_clean_sets()
        data, targets = val_set.data, val_set.targets
    bs = len(data) if bs is None else bs
    n_batches = math.ceil(len(data) / bs)
    if 'ConvAutoencoder' in model.__class__.__name__:
        data = data.to('cpu')
        targets = targets.to('cpu')
    else:
        data = data.to(device)
        targets = targets.to(device)

    # evaluate
    model.to(device)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_idx in range(n_batches):
            data_batch = data[batch_idx * bs: batch_idx * bs + bs].to(device)
            targets_batch = targets[batch_idx * bs: batch_idx * bs + bs].to(device)
            outputs = model(data_batch)
            # flatten outputs in case of ConvAE (targets already flat)
            loss = criterion(torch.flatten(outputs, 1), targets_batch)
            val_loss += loss.item()
    return val_loss / n_batches
