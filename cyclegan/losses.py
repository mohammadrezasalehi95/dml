import torch
import torch.nn.functional as F

def cycle_consistency_loss(real_images, generated_images):
    return torch.mean(torch.abs(real_images - generated_images))

def lsgan_loss_generator(prob_fake_is_real):

    return torch.mean((prob_fake_is_real - 1)**2)

def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
    return 0.5 * (torch.mean((prob_real_is_real - 1)**2) + torch.mean(prob_fake_is_real**2))

def _softmax_weighted_loss(logits, gt):
    softmaxpred = F.softmax(logits, dim=1)
    raw_loss = 0
    for i in range(5):
        gti = gt[:,:,:,i]
        predi = softmaxpred[:,:,:,i]
        weighted = 1 - (torch.sum(gti) / torch.sum(gt))
        raw_loss -= weighted * gti * torch.log(torch.clamp(predi, min=0.005, max=1))
    loss = torch.mean(raw_loss)
    return loss

def _dice_loss_fun(logits, gt):
    dice = 0
    eps = 1e-7
    softmaxpred = F.softmax(logits, dim=1)
    for i in range(5):
        inse = torch.sum(softmaxpred[:, :, :, i]*gt[:, :, :, i])
        l = torch.sum(softmaxpred[:, :, :, i]**2)
        r = torch.sum(gt[:, :, :, i])
        dice += 2.0 * inse/(l+r+eps)
    return 1 - dice / 5

def task_loss(prediction, gt):
    ce_loss = _softmax_weighted_loss(prediction, gt)
    dice_loss = _dice_loss_fun(prediction, gt)
    return ce_loss, dice_loss