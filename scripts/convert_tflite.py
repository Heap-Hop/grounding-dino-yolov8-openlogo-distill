from ultralytics import YOLO # type: ignore
import argparse

# https://docs.ultralytics.com/integrations/tflite/#usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO model to TFLite format.")
    parser.add_argument("--model_path", "-m", type=str, required=True, help="Path to the YOLO model file (e.g., yolov8n.pt).")
    args = parser.parse_args()

    # Load the YOLO model
    model = YOLO(args.model_path)

    # Export the model to TFLite format
    model.export(format="tflite")  # creates '<model_name>_float32.tflite'