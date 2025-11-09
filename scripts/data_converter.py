import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_voc_to_yolo(voc_xml_path, img_width, img_height):
    tree = ET.parse(voc_xml_path)
    root = tree.getroot()
    
    yolo_annotations = []
    
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_annotations.append(f"0 {x_center} {y_center} {width} {height}")
    
    return yolo_annotations


def prepare_dataset(source_dir, output_dir, train_ratio=0.8):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    annotations_dir = source_path / "Annotations"
    images_dir = source_path / "JPEGImages"
    
    train_images_dir = output_path / "images" / "train"
    val_images_dir = output_path / "images" / "val"
    train_labels_dir = output_path / "labels" / "train"
    val_labels_dir = output_path / "labels" / "val"
    
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    xml_files = sorted(list(annotations_dir.glob("*.xml")))
    total_files = len(xml_files)
    train_count = int(total_files * train_ratio)
    
    print(f"Total annotations: {total_files}")
    print(f"Train: {train_count}, Val: {total_files - train_count}")
    
    for idx, xml_file in enumerate(tqdm(xml_files, desc="Converting dataset")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find('filename').text
        if not filename.endswith('.jpg'):
            filename += '.jpg'
        
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        img_path = images_dir / filename
        if not img_path.exists():
            continue
        
        yolo_annotations = convert_voc_to_yolo(xml_file, img_width, img_height)
        
        if not yolo_annotations:
            continue
        
        is_train = idx < train_count
        dest_img_dir = train_images_dir if is_train else val_images_dir
        dest_label_dir = train_labels_dir if is_train else val_labels_dir
        
        shutil.copy(img_path, dest_img_dir / filename)
        
        label_filename = xml_file.stem + '.txt'
        with open(dest_label_dir / label_filename, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
    print("Dataset conversion completed!")


# Example usage:
# python utils/data_converter.py -s /path/to/OpenLogo -o /path/to/YOLODataset -r 0.8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OpenLogo dataset from VOC to YOLO format")
    parser.add_argument("--source_dir", "-s", type=str, required=True, help="Path to the source OpenLogo dataset directory")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Path to the output YOLO formatted dataset directory")
    parser.add_argument("--train_ratio", "-r", type=float, default=0.8, help="Ratio of training data")
    args = parser.parse_args()
    prepare_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio
    )