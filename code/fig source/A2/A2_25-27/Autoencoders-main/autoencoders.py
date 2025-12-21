import math
from abc import abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Sequence, Union, Tuple
from training_utilities import get_clean_sets, get_noisy_sets, fit_ae


# set device globally
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AbstractAutoencoder(nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, mode='basic', tr_data=None, val_data=None, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
        return fit_ae(model=self, mode=mode, tr_data=tr_data, val_data=val_data, num_epochs=num_epochs, bs=bs, lr=lr,
                      momentum=momentum, **kwargs)

    def show_manifold_convergence(self, load=None, path=None, max_iters=1000, thresh=0.02, side_len=28, save=False):
        """
        Show the manifold convergence of an AE when fed with random noise.
        The output of the AE is fed again as input in an iterative process.
        :param load: if True, load an images progression of the manifold convergence
        :param path: path of the images progression
        :param max_iters: max number of iterations.
        :param thresh: threshold of MSE between 2 iterations under which the process is stopped
        :param side_len: length of the side of the images
        :param save: if True, save the images progression and the animation
        """
        if load:
            images_progression = np.load(path)
        else:
            self.cpu()
            noise_img = torch.randn((1, 1, side_len, side_len))
            noise_img -= torch.min(noise_img)
            noise_img /= torch.max(noise_img)
            images_progression = [torch.squeeze(noise_img)]
            serializable_progression = [torch.squeeze(noise_img).cpu().numpy()]

            # iterate
            i = 0
            loss = 1000
            input = noise_img
            prev_output = None
            with torch.no_grad():
                while loss > thresh and i < max_iters:
                    output = self(input)
                    img = torch.reshape(torch.squeeze(output), shape=(side_len, side_len))
                    rescaled_img = (img - torch.min(img)) / torch.max(img)
                    images_progression.append(rescaled_img)
                    serializable_progression.append(rescaled_img.cpu().numpy())
                    if prev_output is not None:
                        loss = F.mse_loss(output, prev_output)
                    prev_output = output
                    input = output
                    i += 1

            # save sequence of images
            if save:
                serializable_progression = np.array(serializable_progression)
                np.save(file="manifold_img_seq", arr=serializable_progression)

        if save:
            images_progression = images_progression[:60]
            frames = []  # for storing the generated images
            fig = plt.figure()
            for i in range(len(images_progression)):
                frames.append([plt.imshow(images_progression[i], animated=True)])
            ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
            ani.save('movie.gif')
            plt.show()
        else:
            # show images progression
            img = None
            for i in range(len(images_progression)):
                if img is None:
                    img = plt.imshow(images_progression[0])
                else:
                    img.set_data(images_progression[i])
                plt.pause(.1)
                plt.draw()


class ShallowAutoencoder(AbstractAutoencoder):
    """ Standard shallow AE with 1 fully-connected layer in the encoder and 1 in the decoder """
    def __init__(self, input_dim: int = 784, latent_dim: int = 200, use_bias=True):
        super().__init__()
        assert input_dim > 0 and latent_dim > 0
        self.type = "shallowAE"
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, latent_dim, bias=use_bias), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim, bias=use_bias), nn.Sigmoid())


class DeepAutoencoder(AbstractAutoencoder):
    """ Standard deep AE """
    def __init__(self, dims: Sequence[int], use_bias=True):
        """
        :param dims: seq of integers specifying the dimensions of the layers (length of dims = number of layers)
        :param use_bias: if False, don't use bias
        """
        super().__init__()
        assert len(dims) > 0 and all(d > 0 for d in dims)
        self.type = "deepAE"
        self.use_bias = use_bias
        enc_layers = []
        dec_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1], bias=use_bias))
            enc_layers.append(nn.ReLU(inplace=True))
        for i in reversed(range(1, len(dims))):
            dec_layers.append(nn.Linear(dims[i], dims[i - 1], bias=use_bias))
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers[-1] = nn.Sigmoid()
        self.encoder = nn.Sequential(nn.Flatten(), *enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def pretrain_layers(self, num_epochs, bs, lr, momentum, mode='basic', freeze_enc=False, **kwargs):
        tr_data = None
        val_data = None
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                print(f"Pretrain layer: {layer}")
                # create shallow AE corresponding to the current layer
                shallow_ae = ShallowAutoencoder(layer.in_features, layer.out_features, use_bias=self.use_bias)
                if freeze_enc:
                    # freeze shallow encoder's weights in case of randomized AE
                    shallow_ae.encoder[1].weight.requires_grad = False
                # train the shallow AE
                shallow_ae.fit(mode=mode, tr_data=tr_data, val_data=val_data, num_epochs=num_epochs, bs=bs, lr=lr,
                               momentum=momentum, **kwargs)
                if freeze_enc:
                    # in case of rand AE, copy shallow decoder's weights transpose in the shallow encoder.
                    # This way it's possible to just copy the weights into the original model without further actions
                    shallow_ae.encoder[1].weight = nn.Parameter(shallow_ae.decoder[0].weight.T)
                # copy shallow AE's weights into the original bigger model
                self.encoder[i].weight = nn.Parameter(shallow_ae.encoder[1].weight)
                self.decoder[len(self.decoder) - i - 1].weight = nn.Parameter(shallow_ae.decoder[0].weight)
                if self.use_bias:
                    self.encoder[i].bias = nn.Parameter(shallow_ae.encoder[1].bias)
                    self.decoder[len(self.decoder) - i - 1].bias = nn.Parameter(shallow_ae.decoder[0].bias)
                # create training set for the next layer
                if i == 1 and mode == 'denoising':  # i = 1 --> fist Linear layer
                    tr_set, val_set = get_noisy_sets(**kwargs)
                    tr_data, tr_targets = tr_set.data, tr_set
                    val_data, val_targets = val_set.data, val_set.targets
                    mode = 'basic'  # for the pretraining of the deeper layers
                tr_data, val_data = self.create_next_layer_sets(shallow_ae=shallow_ae,
                                                                prev_tr_data=tr_data,
                                                                prev_val_data=val_data)
                if num_epochs // 2 > 10:
                    num_epochs = num_epochs // 2

    @staticmethod
    def create_next_layer_sets(shallow_ae, prev_tr_data=None, prev_val_data=None, unsqueeze=True):
        """ Create training data for the next layer during a layer-wise pretraining """
        train_set, val_set = get_clean_sets()
        prev_tr_data = train_set.data if prev_tr_data is None else prev_tr_data
        prev_val_data = val_set.data if prev_val_data is None else prev_val_data
        with torch.no_grad():
            next_tr_data = torch.sigmoid(shallow_ae.encoder(prev_tr_data))
            next_val_data = torch.sigmoid(shallow_ae.encoder(prev_val_data))
            if unsqueeze:
                next_tr_data, next_val_data = torch.unsqueeze(next_tr_data, 1), torch.unsqueeze(next_val_data, 1)
        return next_tr_data, next_val_data


class DeepRandomizedAutoencoder(DeepAutoencoder):
    def __init__(self, dims: Sequence[int]):
        super().__init__(dims=dims, use_bias=False)
        self.type = "deepRandAE"

    def fit(self, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
        """
        The training of this model is a pretraining of its layers where in the corresponding shallow AE
        only its decoder's weights are trained, the encoder's ones are fixed.
        Then copy the shallow decoder's weights in the corresponding layer of the bigger decoder
        and its transpose in the corresponding layer of the bigger encoder.
        """
        assert 0 < lr < 1 and num_epochs > 0 and bs > 0 and 0 <= momentum < 1
        self.pretrain_layers(num_epochs=num_epochs, bs=bs, lr=lr, momentum=momentum, freeze_enc=True)


class ShallowConvAutoencoder(AbstractAutoencoder):
    """ Convolutional AE with 1 conv layer in the encoder and 1 in the decoder """
    def __init__(self, channels=1, n_filters=10, kernel_size: int = 3, central_dim=100,
                 inp_side_len: Union[int, Tuple[int, int]] = 28):
        super().__init__()
        self.type = "shallowConvAE"
        pad = (kernel_size - 1) // 2  # pad to keep the original area after convolution
        central_side_len = math.floor(inp_side_len / 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=central_side_len ** 2 * n_filters, out_features=central_dim),
            nn.ReLU(inplace=True))

        # set kernel size, padding and stride to get the correct output shape
        kersize = 2 if central_side_len * 2 == inp_side_len else 3
        self.decoder = nn.Sequential(
            nn.Linear(in_features=central_dim, out_features=central_side_len ** 2 * n_filters),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(n_filters, central_side_len, central_side_len)),
            nn.ConvTranspose2d(in_channels=n_filters, out_channels=channels, kernel_size=kersize, stride=2, padding=0),
            nn.Sigmoid())


class DeepConvAutoencoder(AbstractAutoencoder):
    """ Conv Ae with variable number of conv layers """
    def __init__(self, inp_side_len=28, dims: Sequence[int] = (5, 10),
                 kernel_sizes: int = 3, central_dim=100, pool=True):
        super().__init__()
        self.type = "deepConvAE"

        # initial checks
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(dims)
        assert len(kernel_sizes) == len(dims) and all(size > 0 for size in kernel_sizes)

        # build encoder
        step_pool = 1 if len(dims) < 3 else (2 if len(dims) < 6 else 3)
        side_len = inp_side_len
        side_lengths = [side_len]
        dims = (1, *dims)
        enc_layers = []
        for i in range(len(dims) - 1):
            pad = (kernel_sizes[i] - 1) // 2
            enc_layers.append(nn.Conv2d(in_channels=dims[i], out_channels=dims[i + 1], kernel_size=kernel_sizes[i],
                                        padding=pad, stride=1))
            enc_layers.append(nn.ReLU(inplace=True))
            if pool and (i % step_pool == 0 or i == len(dims) - 1) and side_len > 3:
                enc_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                side_len = math.floor(side_len / 2)
                side_lengths.append(side_len)

        # fully connected layers in the center of the autoencoder to reduce dimensionality
        fc_dims = (side_len ** 2 * dims[-1], side_len ** 2 * dims[-1] // 2, central_dim)
        self.encoder = nn.Sequential(
            *enc_layers,
            nn.Flatten(),
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims[1], fc_dims[2]),
            nn.ReLU(inplace=True)
        )

        # build decoder
        central_side_len = side_lengths.pop(-1)
        # side_lengths = side_lengths[:-1]
        dec_layers = []
        for i in reversed(range(1, len(dims))):
            # set kernel size, padding and stride to get the correct output shape
            kersize = 2 if len(side_lengths) > 0 and side_len * 2 == side_lengths.pop(-1) else 3
            pad, stride = (1, 1) if side_len == inp_side_len else (0, 2)
            # create transpose convolution layer
            dec_layers.append(nn.ConvTranspose2d(in_channels=dims[i], out_channels=dims[i - 1], kernel_size=kersize,
                                                 padding=pad, stride=stride))
            side_len = side_len if pad == 1 else (side_len * 2 if kersize == 2 else side_len * 2 + 1)
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers[-1] = nn.Sigmoid()

        self.decoder = nn.Sequential(
            nn.Linear(fc_dims[2], fc_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims[1], fc_dims[0]),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(dims[-1], central_side_len, central_side_len)),
            *dec_layers,
        )
