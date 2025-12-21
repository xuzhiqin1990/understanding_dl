import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        assert input_dim > 0 and latent_dim > 0
        self.flat = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid())

    def forward(self, x):
        x = self.flat(x)
        return self.decoder(self.encoder(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)
        return loss


if __name__ == '__main__':
    # data
    dataset = MNIST('../MNIST/', train=True, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [50000, 10000])

    train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(mnist_val, batch_size=32, num_workers=3, pin_memory=True)

    # ae
    model = LitAutoEncoder(input_dim=28*28, latent_dim=200)

    # training
    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=2)
    trainer.fit(model, train_loader, val_loader)
    # print(trainer.callback_metrics)
