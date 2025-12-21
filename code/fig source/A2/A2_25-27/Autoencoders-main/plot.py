import matplotlib.pyplot as plt
import scienceplots
import numpy as np
plt.style.use('science')
x = np.load('/home/linpengxiao/hlk/Autoencoders-main/loss2.npy',allow_pickle=True).item()
print(x)
train_loss = x['tr_loss']
valid_loss = x['val_loss']
with plt.style.context('science'):
    plt.figure()
    plt.plot(list(range(1, len(train_loss)+1)), train_loss, label='train')
    plt.plot(list(range(1, len(train_loss)+1)), valid_loss, label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    plt.savefig('loss.png')