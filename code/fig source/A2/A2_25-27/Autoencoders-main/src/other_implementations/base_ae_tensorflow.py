import time
from abc import ABC
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.models import Model


class Autoencoder(ABC, Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu', input_dim=784),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == '__main__':
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    latent_dim = 200
    ae = Autoencoder(latent_dim)
    start = time.time()
    sgd = optimizers.SGD(learning_rate=0.1, momentum=0.6)
    ae.compile(optimizer=sgd, loss='mean_squared_error')
    ae.fit(x=x_train, y=x_train, batch_size=32, epochs=30, validation_data=(x_test, x_test), shuffle=True)
    print(f"Execution time: {time.time() - start}")
    # ae.save('../models/base_ae_TF', save_format='tf')

    # print the first reconstructions
    # ae = tf.keras.models.load_model('../models/base_ae_TF')
    # for i, img in enumerate(x_test):
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(img)
    #     reconstruction = ae.predict(x_test[i: i + 1])
    #     ax[1].imshow(reconstruction[0, :, :])
    #     plt.show()
    #     if i >= 6:
    #         break
