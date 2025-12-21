import matplotlib.pyplot as plt
from training_utilities import get_clean_sets
from torch.utils import data
import torch

mnist_train, mnist_valid = get_clean_sets()
train_iter=data.DataLoader(mnist_train, batch_size=1, shuffle=False)
model = torch.load('/home/linpengxiao/hlk/Autoencoders-main/AE_adam.pth')


for i in range(1):
    image, target,label = next(iter(train_iter))

    plt.figure()
    plt.imshow(image[0].to('cpu').squeeze(), cmap="gray") 
    plt.savefig('./pic.png', dpi=400)

    hidden_state = model.encoder(image.to('cuda:7'))
    noise = 5*torch.randn(hidden_state.shape, device=hidden_state.device)
    hidden_state = hidden_state + noise
    print(torch.sqrt(torch.sum(noise**2)) / hidden_state.shape[-1])
    
    print('hidden:', hidden_state)
    print('hidden:', hidden_state.shape)
    print('hidden-norm:', torch.sqrt(torch.sum(hidden_state**2)) / hidden_state.shape[-1])
    output = model.decoder(hidden_state)



    plt.figure()
    plt.imshow(output[0].reshape(28,28).detach().to('cpu').squeeze(), cmap="gray") 
    plt.savefig('./pic2.png', dpi=400) 

    plt.figure()
    plt.imshow(image[0].reshape(28,28).detach().to('cpu').squeeze(), cmap="gray") 
    plt.savefig('./pic3.png', dpi=400) 


    		# 显示批量中的图片[0]
# print('label:', label[0]) 
# print('target:', target[0]) 
# print('data:', image[0])  