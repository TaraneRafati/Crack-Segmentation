import os, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
from model import CrackSegNet, CrackSegNet_Dropout, CrackSegNet_Transfer
from tqdm import tqdm

def infer_and_save_images(cfg, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = cfg["infer_input_dir"]
    base_output_dir = cfg["infer_output_dir"]
    input_size = cfg.get("input_size", (512, 512))

    plot_dir = os.path.join(base_output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    use_dropout = cfg.get("use_dropout", False)
    use_transfer_learning = cfg.get("use_transfer_learning", False)
    
    if use_dropout:
        model = CrackSegNet_Dropout().to(device)
    elif use_transfer_learning:
        model = CrackSegNet_Transfer().to(device)
    else:
        model = CrackSegNet().to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_paths:
        print(f"No images found in '{input_dir}'. Please check your config.json and directory path.")
        return

    print(f"Starting inference on {len(image_paths)} images from '{input_dir}'...")
    for img_path in tqdm(image_paths, desc="Inferring images"):
        img_name = os.path.basename(img_path)
        
        orig_img = cv2.imread(img_path)
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_shape = orig_img.shape
        
        img = cv2.resize(orig_img, input_size).astype("float32")/255.0
        img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img)
            pred = torch.sigmoid(out).cpu().squeeze().numpy()
            
        mask = (pred > 0.5).astype("uint8")
        mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        masked_pred = np.ma.masked_where(mask == 0, mask)
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        axs[0].imshow(orig_img_rgb)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        
        axs[1].imshow(orig_img_rgb)
        axs[1].imshow(masked_pred, cmap="jet", alpha=0.5, interpolation='none')
        axs[1].set_title("Predicted Overlay")
        axs[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{os.path.splitext(img_name)[0]}_plot.png"), bbox_inches='tight')
        plt.close(fig)
    
    print("Inference complete. Plots saved to:", plot_dir)