from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

TRAIN_CONFIG = {
    "model": "yolov8n.pt",
    "data": str(Path(__file__).parent / "dataset_gdino.yaml"),
    "epochs": 32,
    "imgsz": 640,
    "batch": 16,
    "workers": 8,
    "device": 0,
    "project": str(PROJECT_ROOT / "output-yolov8-openlogo-distill-gdino"),
    "name": "logo_detection",
    "patience": 20,
    "save": True,
    "save_period": 10,
    "cache": False,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
}

EVAL_CONFIG = {
    "imgsz": 640,
    "batch": 32,
    "device": 0,
    "workers": 8,
    "verbose": True,
}
