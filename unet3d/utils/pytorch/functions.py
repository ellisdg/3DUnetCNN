from torch.nn.functional import l1_loss, mse_loss
import torch
from .pt3dunet import compute_per_channel_dice


def per_channel_dice_loss(x, y, **kwargs):
    return 1 - compute_per_channel_dice(x, y, **kwargs).mean()


def variational_regularized_loss(predicted, vae_x, mu, logvar, x, y, pred_loss=l1_loss, decoder_loss=mse_loss,
                                 vae_weight=0.1, kl_weight=0.1):
    loss_pred = pred_loss(predicted, y)
    loss_vae = decoder_loss(vae_x, x)
    N = x.numel()/x.shape[0]
    loss_kl = (1 / N) * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    return loss_pred + (vae_weight * loss_vae) + (kl_weight * loss_kl)


def regularized_loss(predicted_y, predicted_x, x, y, pred_loss=l1_loss, decoder_loss=mse_loss, decoder_weight=0.1):
    return pred_loss(predicted_y, y) + decoder_weight * decoder_loss(predicted_x, x)


def kl_loss(mu, logvar, N):
    return (1 / N) * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)


def vae_loss(predicted_x, mu, logvar, x, recon_loss=mse_loss, divergence_loss=kl_loss, recon_weight=1, kl_weight=1):
    loss_recon = recon_loss(predicted_x, x)
    loss_kl = divergence_loss(mu, logvar, x.numel()/x.shape[0])
    return recon_weight * loss_recon + kl_weight * loss_kl


def dice_loss(input, target, smooth=1.):
    iflat = input.view(-1).float()
    tflat = target.view(-1).float()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))


def vae_dice_loss(predicted, mu, logvar, target, loss=dice_loss, divergence_loss=kl_loss, weight=1, kl_weight=1):
    return vae_loss(predicted_x=predicted, mu=mu, logvar=logvar, x=target, recon_loss=loss,
                    divergence_loss=divergence_loss, recon_weight=weight, kl_weight=kl_weight)


def vae_l1_loss(predicted, mu, logvar, target, loss=l1_loss, divergence_loss=kl_loss, weight=1, kl_weight=1):
    return vae_loss(predicted_x=predicted, mu=mu, logvar=logvar, x=target, recon_loss=loss,
                    divergence_loss=divergence_loss, recon_weight=weight, kl_weight=kl_weight)


def weighted_loss(input, target, weights, loss_func, weighted_dimension=1):
    losses = torch.zeros(input.shape[weighted_dimension])
    for index in range(input.shape[weighted_dimension]):
        x = input.select(dim=weighted_dimension, index=index)
        y = target.select(dim=weighted_dimension, index=index)
        losses[index] = loss_func(x, y)
    return torch.mean(weights * losses)


class WeightedLoss(object):
    def __init__(self, weights, loss_func, weighted_dimension=1):
        self.weights = weights
        self.loss_func = loss_func
        self.weighted_dimension = weighted_dimension

    def __call__(self, input, target):
        return weighted_loss(input=input, target=target, weights=self.weights, loss_func=self.loss_func,
                             weighted_dimension=self.weighted_dimension)
