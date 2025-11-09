import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

def evaluate_model(model_path: str, data_yaml: str, device: str = '0'):
    """Evaluate a YOLO model on validation set."""
    model = YOLO(model_path)
    
    # Run validation
    print(f"\nEvaluating {model_path}...")
    results = model.val(data=data_yaml, device=device, verbose=False)
    
    return {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr
    }

def create_comparison_table(results_dict: dict, output_path: str):
    """Create comparison table and save as markdown."""
    df = pd.DataFrame(results_dict).T
    df.index.name = 'Model'
    
    # Round values for better display
    df['mAP50'] = (df['mAP50'] * 100).round(2)
    df['mAP50-95'] = (df['mAP50-95'] * 100).round(2)
    df['precision'] = (df['precision'] * 100).round(2)
    df['recall'] = (df['recall'] * 100).round(2)
    
    # Rename columns for better display
    df.columns = ['mAP@0.5 (%)', 'mAP@0.5:0.95 (%)', 'Precision (%)', 'Recall (%)']
    
    print("\n" + "="*80)
    print("Model Comparison Results")
    print("="*80)
    print(df.to_string())
    print("="*80 + "\n")
    
    # Save as markdown
    with open(output_path, 'w') as f:
        f.write("## Model Comparison Results\n\n")
        f.write(df.to_markdown())
        f.write("\n")
    
    return df

def create_comparison_charts(df: pd.DataFrame, output_dir: Path):
    """Create comparison chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Chart: Accuracy Metrics (mAP, Precision, Recall)
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['mAP@0.5 (%)', 'mAP@0.5:0.95 (%)', 'Precision (%)', 'Recall (%)']
    x = np.arange(len(metrics))
    width = 0.35
    
    models = df.index.tolist()
    for i, model in enumerate(models):
        values = [df.loc[model, metric] for metric in metrics]
        ax.bar(x + i*width, values, width, label=model)
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300)
    plt.close()
    
    print(f"Chart saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Compare two YOLO models')
    parser.add_argument('--model1', type=str, 
                        default='output-yolov8-openlogo-human/logo_detection/weights/epoch30.pt',
                        help='Path to first model')
    parser.add_argument('--model2', type=str,
                        default='output-yolov8-openlogo-distill-gdino/logo_detection/weights/epoch30.pt',
                        help='Path to second model')
    parser.add_argument('--data', type=str,
                        default='scripts/config/dataset_human.yaml',
                        help='Path to dataset yaml')
    parser.add_argument('--output-dir', type=str,
                        default='comparison_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (0 for GPU, cpu for CPU)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate both models
    results = {}
    results['Human Labels'] = evaluate_model(args.model1, args.data, args.device)
    results['GDINO Distilled'] = evaluate_model(args.model2, args.data, args.device)
    
    # Create comparison table
    df = create_comparison_table(results, str(output_dir / 'comparison_table.md'))
    
    # Create charts
    create_comparison_charts(df, output_dir)
    
    # Save raw results as JSON
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll results saved to {output_dir}")

if __name__ == '__main__':
    main()
