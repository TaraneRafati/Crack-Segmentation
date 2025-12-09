import torch
from metrics import iou_score, dice_score

def evaluate(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ious, dices = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.unsqueeze(1).to(device)  
            out = model(imgs)
            ious.append(iou_score(out, masks))
            dices.append(dice_score(out, masks))
    print(f"Mean IoU: {sum(ious)/len(ious):.3f}, Dice: {sum(dices)/len(dices):.3f}")