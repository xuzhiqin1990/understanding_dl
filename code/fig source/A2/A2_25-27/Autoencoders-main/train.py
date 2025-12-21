from training_utilities import fit_ae, evaluate
from autoencoders import ShallowAutoencoder
import numpy as np
def main():

    model = ShallowAutoencoder()
    hist = fit_ae(model=model, mode='basic', num_epochs=100)
    np.save('loss2.npy', hist)


if __name__ == '__main__':
    main()
    

