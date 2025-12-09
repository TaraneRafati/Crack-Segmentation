import argparse, torch, os
from dataloader import get_loader
from model import CrackSegNet, CrackSegNet_Dropout, CrackSegNet_Transfer
from train import train_model
from evaluate import evaluate
from utils import load_config
from utils import plot_loss_curves, test_dataloader, test_augmented_dataloader, visualize_predictions
from inference import infer_and_save_images
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","eval","infer"], required=True)
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_path = cfg["data_path"]
    vis_dir = cfg.get("vis_dir")
    use_dropout = cfg.get("use_dropout", False)
    use_transfer_learning = cfg.get("use_transfer_learning", False)

    if args.mode=="train":
        train_loader = get_loader(cfg["train_json_path"], data_path, augment=cfg["augment"])
        val_loader   = get_loader(cfg["val_json_path"], data_path, shuffle=False)
        if use_dropout:
            model = CrackSegNet_Dropout()
        elif use_transfer_learning:
            model = CrackSegNet_Transfer(encoder_name=cfg.get("transfer_encoder"))
        else:
            model = CrackSegNet()
            
        history = train_model(
            model, 
            train_loader, 
            val_loader, 
            epochs=cfg["epochs"],
            lr=cfg["learning_rate"],
            save_path=cfg["save_model_path"]
        )
        
        plot_loss_curves(history, cfg.get("loss_path"))

        print("Training finished. Starting evaluation on the test set...")
        model.load_state_dict(torch.load(cfg["save_model_path"]))
        test_loader = get_loader(cfg["test_json_path"], data_path, shuffle=False)
        evaluate(model, test_loader)

        print(f"Saving a few predictions to '{vis_dir}'...")
        visualize_predictions(model, test_loader, num_samples=cfg.get("num_samples_vis", 5), save_dir=vis_dir)

    elif args.mode=="eval":
        test_loader = get_loader(cfg["test_json_path"], data_path, shuffle=False)
        if use_dropout:
            model = CrackSegNet_Dropout()
        elif use_transfer_learning:
            model = CrackSegNet_Transfer()
        else:
            model = CrackSegNet()
        model.load_state_dict(torch.load(cfg["model_path"]))
        evaluate(model, test_loader)

    elif args.mode=="infer":
        infer_and_save_images(cfg, cfg["model_path"])

if __name__=="__main__":
    # task : test dataloader
    # loader = get_loader("jsons/train_70.coco.json", "data", batch_size=1, shuffle=True)
    # test_dataloader(loader, num_samples= 6)

    # task : test augmented dataloader
    # augment_loader = get_loader("jsons/train_70.coco.json", "data", batch_size=1, shuffle=True, augment=True, aug_factor=5)
    # test_augmented_dataloader(augment_loader, num_samples=1)

    main()
