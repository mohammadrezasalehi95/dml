
from torch import Tensor
import torch
from torch.nn import functional as F
import numpy as np

from tqdm import tqdm
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', position=0,leave=False):
            image, mask_true,labels = batch
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            labels=labels.to(device=device,dtype=torch.float32 )
            # predict the mask
            mask_pred,labels_pred = net(image)
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)

def evaluate2():
        self._tp_per_class: List[float] = np.zeros((self._c,), dtype=int)
        self._fp_per_class: List[float] = np.zeros_like(self._tp_per_class)
        self._fn_per_class: List[float] = np.zeros_like(self._tp_per_class)


        # fill this 
        preds = self._read_model_seg_prediction(model_output)
        ground_truths = self._read_gt_seg_prediction(model_output)
       
        # assuming 0 as background
        for c in range(1, self._c):
            preds_c = (preds == c).astype(int)
            ground_truths_c = (ground_truths == c).astype(int)

            for gt, pd in zip(ground_truths_c, preds_c):

                self._n_received_samples += 1
 
                gt_labels = measure.label(gt)
                pd_labels = measure.label(pd)

                for l in np.unique(gt_labels):
                    if l == 0:
                        continue

                    # if ground_truth obj has enough overlap with pred, it's captured
                    gt_obj_mask = (gt_labels == l)

                    # if l has any overlap with check mask, it's captured
                    if np.sum(
                            (gt_obj_mask & (pd == 1))) \
                            * 1.0 / np.sum(gt_obj_mask) >= \
                            tp_fn_threshold:
                        self._tp_per_class[c] += 1
                    else:
                        self._fn_per_class[c] += 1
               
                for l in np.unique(pd_labels):
                    if l == 0:
                        continue

                    pd_obj_mask = (pd_labels == l)

                    # if l has any overlap with check mask, it's captured
                    if 1.0 * np.sum(pd_obj_mask &
                                (gt == 1)) / \
                            np.sum(pd_obj_mask) < self._threshold:
                        self._fp_per_class[c] += 1

