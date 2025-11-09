### Prerequisites

#### Convert OpenLogo dataset to YOLO format

```
uv run scripts/data_converter.py -s dataset/openlogo -o dataset/openlogo_yolo
```

#### Download GroundingDINO weights
TODO

#### Generate Pseudo-Labels using GroundingDINO


```
cd GroundingDINO
```

##### Build GroundingDINO

[set CUDA_HOME](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install)

```
export CUDA_HOME=/usr/local/cuda  
uv pip install -e . --no-build-isolation
```

##### Run in test mode

```bash
uv run generate_gdino_labels.py \
  --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
  --checkpoint_path weights/groundingdino_swinb_cogcoor.pth \
  --dataset_path ../dataset/openlogo_yolo \
  --output_path ../dataset/gdino_distill \
  --text_prompt "logo" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --min_size 0.001 \
  --max_size 0.8 \
  --max_aspect_ratio 5.0 \
  --test_mode \
  --test_samples 10
```
you can check the generated labels in `../dataset/gdino_distill/labels` and corresponding images in `../dataset/gdino_distill/visualizations`.

##### Remove test results
```
rm -r ../dataset/gdino_distill/labels
```

##### Generate pseudo-labels for the entire dataset

```bash
uv run generate_gdino_labels.py \
  --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
  --checkpoint_path weights/groundingdino_swinb_cogcoor.pth \
  --dataset_path ../dataset/openlogo_yolo \
  --output_path ../dataset/gdino_distill \
  --text_prompt "logo" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --min_size 0.001 \
  --max_size 0.8 \
  --max_aspect_ratio 5.0
```

##### Create soft link for images
Make sure we will use the separate folder for distilled labels and images.

(go back to root folder)
```bash
cd ..
```

```bash
ln -s "$(pwd)/dataset/openlogo_yolo/images" dataset/gdino_distill/
```

### Train YOLOv8 with original labels

```bash
uv run scripts/train.py --supervisor
```

### Train YOLOv8 with distilled labels

```bash
uv run scripts/train.py --gdino_distill
```
