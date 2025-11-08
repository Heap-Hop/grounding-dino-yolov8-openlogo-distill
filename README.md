### Prerequisites

#### Convert OpenLogo dataset to YOLO format

```
uv run utils/data_converter.py -s dataset/openlogo -o dataset/openlogo_yolo
```

#### Download GroundingDINO weights
TODO

#### Generate Pseudo-Labels using GroundingDINO


```
cd GroundingDINO
```

build GroundingDINO
```
uv pip install -e . --no-build-isolation
```

test mode
```
uv run generate_gdino_labels.py \
  --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
  --checkpoint_path weights/groundingdino_swinb_cogcoor.pth \
  --dataset_path ../dataset/openlogo_yolo \
  --output_path ../dataset/openlogo_yolo/labels_gdino \
  --text_prompt "logo" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --min_size 0.001 \
  --max_size 0.8 \
  --max_aspect_ratio 5.0 \
  --test_mode \
  --test_samples 10
```

generate pseudo-labels for the entire dataset

```
uv run generate_gdino_labels.py \
  --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
  --checkpoint_path weights/groundingdino_swinb_cogcoor.pth \
  --dataset_path ../dataset/openlogo_yolo \
  --output_path ../dataset/openlogo_yolo/labels_gdino \
  --text_prompt "logo" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --min_size 0.001 \
  --max_size 0.8 \
  --max_aspect_ratio 5.0
```