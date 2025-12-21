import torch
from torch import Tensor
import torch.nn.modules.loss as loss


class ContractiveLoss(loss.MSELoss):
    """
    Custom loss for contractive autoencoders.

    note: the superclass is MSELoss, simply because the base class _Loss is protected and it's not a best practice.
          there isn't a real reason between the choice of MSELoss, since the forward method is overridden completely.

    Overridden for elasticity -> it's possible to use a function as a custom loss, but having a wrapper class
    allows to do:
        criterion = ClassOfWhateverLoss()
        loss = criterion(output, target)    # this line always the same regardless of the type on loss
    """
    def __init__(self, ae, lambd: float, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(ContractiveLoss, self).__init__(size_average, reduce, reduction)
        self.ae = ae
        self.lambd = lambd

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return contractive_loss(input, target, self.lambd, self.ae, self.reduction)


def contractive_loss(input, target, lambd, ae, reduction: str):
    """
    Actual function computing the loss of a contractive autoencoder
    :param input: (Tensor)
    :param target: (Tensor)
    :param lambd: (float) regularization parameter
    :param ae: (DeepAutoencoder) the model itself, used to get it's weights
    :param reduction: (str) type of reduction {'mean' | 'sum'}
    :raises: ValueError
    :return: the loss
    """
    term1 = (input - target) ** 2
    enc_weights = [ae.encoder[i].weight for i in reversed(range(1, len(ae.encoder), 2))]
    term2 = lambd * torch.norm(torch.chain_matmul(*enc_weights))
    contr_loss = torch.mean(term1 + term2, 0)
    if reduction == 'mean':
        return torch.mean(contr_loss)
    elif reduction == 'sum':
        return torch.sum(contr_loss)
    else:
        raise ValueError(f"value for 'reduction' must be 'mean' or 'sum', got {reduction}")
