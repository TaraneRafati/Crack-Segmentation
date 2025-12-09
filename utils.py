import json
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch


def load_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def test_dataloader(loader, num_samples=3):

    for i, (img, mask) in enumerate(loader):
        print(f"Sample {i+1}: image shape = {img.shape}, mask shape = {mask.shape}")

        img_np = img[0].permute(1,2,0).numpy()
        mask_np = mask[0].numpy()

        if img_np.max() > 1.0:
            img_np = img_np.astype(np.uint8)

        fig, axs = plt.subplots(1, 3, figsize=(12,4))
        axs[0].imshow(img_np)
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(mask_np, cmap="gray")
        axs[1].set_title("Mask")
        axs[1].axis("off")

        masked_mask = np.ma.masked_where(mask_np == 0, mask_np)
        axs[2].imshow(img_np)
        axs[2].imshow(masked_mask, cmap="jet", alpha=0.5, interpolation='none')
        axs[2].set_title("Overlay")
        axs[2].axis("off")

        plt.show()

        if i+1 >= num_samples:
            break


def test_augmented_dataloader(loader, num_samples=3):
    count = 0
    for i, (imgs, masks) in enumerate(loader):
        print(f"Batch {i+1}: image shape = {imgs.shape}, mask shape = {masks.shape}")

        for j in range(imgs.shape[0]):
            img_np = imgs[j].permute(1,2,0).numpy()
            mask_np = masks[j].numpy()
            
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].imshow(img_np)
            axs[0].set_title("Image")
            axs[0].axis("off")

            axs[1].imshow(mask_np, cmap="gray")
            axs[1].set_title("Mask")
            axs[1].axis("off")

            masked_mask = np.ma.masked_where(mask_np == 0, mask_np)
            axs[2].imshow(img_np)
            axs[2].imshow(masked_mask, cmap="jet", alpha=0.5, interpolation='none')
            axs[2].set_title("Overlay")
            axs[2].axis("off")

            plt.show()
            
            count += 1
            if count >= num_samples:
                return
            
def visualize_predictions(model, loader, num_samples=3, save_dir="predictions"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.unsqueeze(1).to(device)
            outputs = model(imgs)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu()

            for i in range(imgs.shape[0]):
                img_np = imgs[i].cpu().permute(1,2,0).numpy()
                mask_np = masks[i].cpu().squeeze().numpy()
                pred_np = preds[i].squeeze().numpy()

                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                
                axs[0].imshow(img_np)
                axs[0].set_title("Image")
                axs[0].axis("off")

                axs[1].imshow(mask_np, cmap="gray")
                axs[1].set_title("True Mask")
                axs[1].axis("off")

                axs[2].imshow(pred_np, cmap="gray")
                axs[2].set_title("Predicted Mask")
                axs[2].axis("off")
                
                masked_pred = np.ma.masked_where(pred_np == 0, pred_np)
                axs[3].imshow(img_np)
                axs[3].imshow(masked_pred, cmap="jet", alpha=0.5, interpolation='none')
                axs[3].set_title("Overlay Prediction")
                axs[3].axis("off")

                plt.savefig(os.path.join(save_dir, f"prediction_{count}.png"))
                plt.close(fig) 
                
                count += 1
                if count >= num_samples:
                    return
                
def random_rotation(img, mask, angle_range=(-30, 30)):
    angle = np.random.uniform(*angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img_rot = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask_rot = cv2.warpAffine(mask, M, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return img_rot, mask_rot

def random_scaling(img, mask, scale_range=(0.8,1.2)):
    scale = np.random.uniform(*scale_range)
    h, w = img.shape[:2]
    new_h, new_w = int(h*scale), int(w*scale)
    img_scaled = cv2.resize(img, (new_w,new_h), interpolation=cv2.INTER_LINEAR)
    mask_scaled = cv2.resize(mask, (new_w,new_h), interpolation=cv2.INTER_NEAREST)

    top = max((new_h - h)//2, 0)
    left = max((new_w - w)//2, 0)
    img_cropped = cv2.resize(img_scaled[top:top+h, left:left+w], (w,h), interpolation=cv2.INTER_LINEAR)
    mask_cropped = cv2.resize(mask_scaled[top:top+h, left:left+w], (w,h), interpolation=cv2.INTER_NEAREST)
    return img_cropped, mask_cropped

def random_brightness(img, mask, brightness_range=(-0.2, 0.2)):
    brightness = np.random.uniform(*brightness_range)
    img_b = img.astype(np.float32)
    img_b = img_b + (brightness * 255)
    img_b = np.clip(img_b, 0, 255).astype(np.uint8)
    return img_b, mask

def random_contrast(img, mask, contrast_range=(-0.2, 0.2)):
    contrast = np.random.uniform(*contrast_range)
    img_c = img.astype(np.float32)
    img_c = img_c * (1 + contrast)
    img_c = np.clip(img_c, 0, 255).astype(np.uint8)
    return img_c, mask

def add_gaussian_noise(img, mask, sigma=10):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img_noisy = img.astype(np.float32) + noise
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
    return img_noisy, mask


def plot_loss_curves(history, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    print(f"Loss plot saved to '{save_dir}/loss_plot.png'")