import platform
import torch
import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional
from skimage import measure
import csv
import os
import glob
import numpy as np
# specify your path here

import re
from torch.nn import functional as F

def get_epoch_from_filename(filename):
    match = re.search(r'.*checkpoint_epoch(\d+).pth', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def load_recentliest_file(g_path):
    path=os_support_path(g_path)
    create_directory_if_not_exists(path)
    files = glob.glob(path+"*.pth")
    if files:
        return max(files, key=os.path.getctime)
    else:
        return None
    
    
def os_support_path(path):
    if platform.system() == 'Windows':
        return  "C:\\Users\\user01\\dml\\"+path.replace("/","\\")
    else:
        return path
def read_csv_list(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        
    return data[1:]


def balanced_focal_cross_entropy_loss(
        probs:torch.Tensor, gt: torch.Tensor, focal_gamma: float = 1,
        ignore_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
        Calculates class balanced cross entropy loss weighted with the style of Focal loss.
        mean_over_classes(1/mean(focal_weights) * (focal_weights * cross_entropy_loss))
        focal_weight: (1 - p_of_the_related_class)^gamma

        Args:
            probs (torch.Tensor): Pr utdeobabilities assigned by the model of shape B C ...
            gt (torch.Tensor): The ground truth containing the number of the class, whether of shape B ... or B C ..., make sure the ... matches in both probs and gt
            focal_gamma (float): The power factor used in the weight of focal loss. Default is one which means the typical balanced cross entropy.
            ignore_mask (torch.Tensor, Optional): If passed, would skip applying loss on the pixels for which the mask is True.

        Returns:
            torch.Tensor: The calculated
    """

    # if channel is one in the pixel probs, convert it to the binary mode!
    if probs.shape[1] == 1:
        probs = torch.cat([1 - probs, probs], dim=1)

    if ignore_mask is not None:
        if len(ignore_mask.shape) == 3:
            ignore_mask = ignore_mask[:, None, ...]

        assert len(ignore_mask.shape) == 4 and ignore_mask.shape[1] == 1, f'Expected ignore mask of shape B 1 H W or B H W but received {ignore_mask.shape}'

        if ignore_mask.shape[-2:] != probs.shape[-2:]:
            ignore_mask = resize_binarize_roi_torch(ignore_mask, probs.shape[-2:])

    assert gt.max() <= probs.shape[1] - 1, f'Expected {probs.shape[1]} classes according to pixel probs\' shape but found {gt.max()} in GT mask'

    if len(gt.shape) == len(probs.shape):
        assert (gt.shape[1] == 1) or (gt.shape[1] == probs.shape[1]), f'Expected the channel dim to have either one channel or the same as probs while it is of shape {gt.shape} and probs is of shape {probs.shape}'
        if gt.shape[1] == 1:
            gt = gt[:, 0, ...]
        else:
            gt = torch.argmax(gt, dim=1)  # transferring one-hot to count data

    assert len(gt.shape) == (len(probs.shape) - 1), f'Expected GT labels to be of shape B ... and pixel probs of shape B C ..., but received {gt.shape} and {probs.shape} instead.'

    # if probabilities and ground truth have different shapes, resize the ground truth
    if probs.shape[-2:] != gt.shape[-2:]:

        # convert gt to one-hot it before interpolation to prevent mistakes
        gt = F.one_hot(gt.long(), probs.shape[1])  # B H W C
        gt = torch.permute(gt, [0, -1, 1, 2])

        # binarize
        gt = resize_binarize_roi_torch(gt.float(), probs.shape[-2:])

        # convert one-hot to numbers with argmax
        gt = torch.argmax(gt, dim=1)  # Eliminating the channel

    gt = gt.long()  # B H W

    # flattening and bringing class channel o index 0
    probs = probs.transpose(0, 1).flatten(1)  # C N
    gt = gt.flatten(0)  # N

    c = probs.shape[0]

    gt_related_prob = probs[gt, torch.arange(gt.shape[0])]

    losses = []
    # changing gt of ignore mask to -1, so it won't bother!
    if ignore_mask is not None:
        ignore_mask = ignore_mask.flatten(0)
        gt = torch.where(ignore_mask >= 0.5, -1, gt)

    for ci in range(c):
        c_probs = gt_related_prob[gt == ci]

        if torch.numel(c_probs) > 0:

            if focal_gamma == 1:
                losses.append(F.binary_cross_entropy(
                    c_probs,
                    torch.ones_like(c_probs)))
            else:
                w = torch.pow(torch.abs(1 - c_probs.detach()), focal_gamma)
                losses.append(
                    (1.0 / (torch.mean(w) + 1e-4)) *
                    F.binary_cross_entropy(
                        c_probs,
                        torch.ones_like(c_probs),
                        weight=w))

    return torch.mean(torch.stack(losses))

# from skimage import measure
class Eval_FP():
    def __init__(self):
        self._threshold=.5
        self._n_received_samples=0
        self._c=2
        self._tp_per_class: List[float] = np.zeros((self._c,), dtype=int)
        self._fp_per_class: List[float] = np.zeros_like(self._tp_per_class)
        self._fn_per_class: List[float] = np.zeros_like(self._tp_per_class)
        self._tp_fn_threshold=0.5
        pass
    def eval(self,prediction,ground_truth):
        # fill this 
        # preds = self._
        # read_model_seg_prediction(model_output)
        # ground_truths = self._read_gt_seg_prediction(model_output)
        preds=prediction
        ground_truths=ground_truth
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
                            (gt_obj_mask & (pd == 1))) * 1.0 / np.sum(gt_obj_mask) >= self._tp_fn_threshold:
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
                
