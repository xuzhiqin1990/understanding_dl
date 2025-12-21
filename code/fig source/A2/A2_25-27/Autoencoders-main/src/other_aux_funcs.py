import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
import seaborn as sns
from sklearn.manifold import TSNE
from autoencoders import AbstractAutoencoder
from training_utilities import get_clean_sets, get_noisy_sets


def tsne(model, n_components=2, noisy=False, save=False, path="../plots/tsne.png", **kwargs):
    """
    Compute t-SNE
    :param model: the model used to encode the data
    :param n_components: dimensionality of the points in the plot (2D / 3D)
    :param noisy: if True, encode noisy data
    :param save: if True, save the plot
    :param path: if 'save', path where to save the plot
    """
    _, mnist_test = get_noisy_sets(**kwargs) if noisy else get_clean_sets()
    with torch.no_grad():
        model.cpu()
        encoded = torch.flatten(model.encoder(mnist_test.data.cpu()), 1)
        embedded = TSNE(n_components=n_components, verbose=1).fit_transform(encoded)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=embedded[:, 0], y=embedded[:, 1],
            hue=mnist_test.labels.cpu(),
            palette=sns.color_palette("hls", 10),
            legend="full",
        )
        plt.tight_layout()
        plt.axis('off')
        if save:
            plt.savefig(path)
        else:
            plt.show()


def classification_test(ae: AbstractAutoencoder, noisy=False, tr_data=None, tr_labels=None, ts_data=None, ts_labels=None, **kwargs):
    """
    Performs a classification based on the encodings of an AE.
    Classifier consists of a single softmax layer.
    Implemented in Keras (unlike the rest of the project that uses Pytorch)
    :param ae: autoencoder used to produce the encodings
    :param noisy: use noisy data if True
    :param tr_data: specific training data to use (optional)
    :param tr_labels: labels for the specific training data to use (optional)
    :param ts_data: specific test data to use (optional)
    :param ts_labels: labels for the specific test data to use (optional)
    :return: the evaluation result of the classifier
    """
    # initial checks and data set
    assert (tr_data is None and tr_labels is None) or (tr_data is not None and tr_labels is not None)
    assert (ts_data is None and ts_labels is None) or (ts_data is not None and ts_labels is not None)
    tr_set, ts_set = get_noisy_sets(**kwargs) if noisy else get_clean_sets()
    if tr_data is None:
        tr_data, tr_labels = tr_set.data, tr_set.labels
    if ts_data is None:
        ts_data, ts_labels = ts_set.data, ts_set.labels

    # pass the data through the encoder
    with torch.no_grad():
        if 'Conv' in ae.type:
            bs = 1000
            n_batches = math.ceil(len(tr_data) / bs)
            tr_data, ts_data = tr_data.cpu(), ts_data.cpu()
            new_tr_data = torch.empty(tr_data.shape[0], ae.encoder[-2].weight.shape[0])
            new_ts_data = torch.empty(ts_data.shape[0], ae.encoder[-2].weight.shape[0])
            for batch_idx in range(n_batches - 1):
                train_data_batch = tr_data[batch_idx * bs: batch_idx * bs + bs].to('cuda:0')
                encoded_batch = ae.encoder(train_data_batch)
                new_tr_data[batch_idx * bs: batch_idx * bs + bs] = encoded_batch
                if batch_idx < math.ceil(len(ts_data) / bs):
                    test_data_batch = ts_data[batch_idx * bs: batch_idx * bs + bs].to('cuda:0')
                    encoded_batch = ae.encoder(test_data_batch)
                    new_ts_data[batch_idx * bs: batch_idx * bs + bs] = encoded_batch
            tr_data = new_tr_data.cpu().numpy()
            ts_data = new_ts_data.cpu().numpy()
        else:
            tr_data = ae.encoder(tr_data).cpu().detach().numpy()
            ts_data = ae.encoder(ts_data).cpu().detach().numpy()
    tr_labels = tf.keras.utils.to_categorical(np.array(tr_labels.cpu()), num_classes=10)
    ts_labels = tf.keras.utils.to_categorical(np.array(ts_labels.cpu()), num_classes=10)

    # create and train a classifier made of one softmax layer
    classifier = tf.keras.Sequential([Dense(input_dim=tr_data.shape[-1], units=10, activation='softmax')])
    classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics='accuracy')
    classifier.fit(x=tr_data, y=tr_labels, batch_size=32, epochs=10, verbose=0)

    # test the classifier
    return classifier.evaluate(x=ts_data, y=ts_labels, batch_size=len(ts_data))
