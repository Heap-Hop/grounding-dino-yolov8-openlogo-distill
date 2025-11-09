import torch
from ultralytics import YOLO # type: ignore
import argparse

def train(config):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    model = YOLO(config["model"])
    
    results = model.train(**config)
    
    print("\nTraining completed!")
    print(f"Best model saved to: {results.save_dir}") # type: ignore
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model with specified config.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--human_labels", action="store_true", help="Training in human_labels mode, use origin openlog dataset")
    group.add_argument("--gdino_distill", action="store_true", help="Training in gdino distill mode, use gdino distilled openlogo dataset")
    args = parser.parse_args()

    if args.human_labels:
        from config.config_human_labels import TRAIN_CONFIG
        train(TRAIN_CONFIG)
    elif args.gdino_distill:
        from config.config_gdino_labels import TRAIN_CONFIG
        train(TRAIN_CONFIG)
    else:
        print("Please specify a valid training mode: '--human_labels' or '--gdino_distill'.")
