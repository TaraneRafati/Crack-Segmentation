import torch
import torch.nn as nn

def iou_score(pred, target, thresh=0.5):
    pred = (torch.sigmoid(pred) > thresh).int()
    target = target.int() 
    inter = (pred & target).sum().float()
    union = (pred | target).sum().float()
    return (inter / union).item() if union>0 else 0

def dice_score(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).int()
    target = target.int() 
    inter = (pred & target).sum().float()
    return (2*inter / (pred.sum()+target.sum()+eps)).item()

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight, size_average=size_average)

    def forward(self, inputs, targets, smooth=1):
       
        bce = self.bce_loss(inputs, targets)

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        dice_loss = 1 - dice

        return bce + dice_loss