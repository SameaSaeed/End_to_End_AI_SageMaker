import os
import sys
import json
import yaml
import logging
import shutil
import subprocess
from pathlib import Path
import argparse
import boto3
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SageMaker paths
MODEL_DIR = '/opt/ml/model'
PROCESSING_INPUT_DIR = '/opt/ml/processing/input'
EVALUATION_OUTPUT_DIR = '/opt/ml/processing/evaluation'
LOCAL_WEIGHTS_DIR = '/opt/ml/processing/output'  # intermediate weights

# S3 upload configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'yolo-wildfire-smoke-detection')
S3_FOLDER = os.getenv('S3_FOLDER', 'wildfire_smoke')

def install_packages():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "torch", "onnx", "onnxruntime"])
        logging.info("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install packages: {e}")
        sys.exit(1)

def create_data_yaml(input_dir: str, yaml_path: str):
    data_conf = {
        'train': str(Path(input_dir) / "train"),
        'val': str(Path(input_dir) / "val"),
        'test': str(Path(input_dir) / "test"),
        'names': ['smoke']
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data_conf, f)
    logging.info(f"Generated data.yaml at {yaml_path}")
    return yaml_path

def train_yolo(model_path: str, data_yaml: str, args):
    from ultralytics import YOLO

    logging.info("Starting YOLO training...")
    model = YOLO(model_path)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.initial_learning_rate,
        lrf=args.final_learning_rate,
        project=LOCAL_WEIGHTS_DIR,
        name='train'
    )

    best_model_path = Path(LOCAL_WEIGHTS_DIR) / 'train' / 'weights' / 'best.pt'
    if not best_model_path.exists():
        logging.error(f"best.pt not found at {best_model_path}")
        raise FileNotFoundError("Training did not produce best.pt")

    os.makedirs(MODEL_DIR, exist_ok=True)
    shutil.copy(best_model_path, Path(MODEL_DIR) / 'best.pt')
    logging.info(f"Copied best.pt to {MODEL_DIR}/best.pt")
    return model

def evaluate_model(model, data_yaml: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    metrics = model.val(data=data_yaml, project=output_dir, name="val-results", split="test")
    metrics_dict = {
        "mAP": metrics.box.map,
        "mAP50": metrics.box.map50,
        "mAP75": metrics.box.map75,
        "mAP_list": metrics.box.maps.tolist(),
        "precision": metrics.box.mp,
        "recall": metrics.box.mr
    }
    metrics_json_path = Path(output_dir) / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f)
    logging.info(f"Evaluation metrics saved at {metrics_json_path}")
    return metrics_json_path

def export_to_onnx(model, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = Path(output_dir) / "model.onnx"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(1, 3, 640, 640, device=device)
    model.model.to(device)
    try:
        torch.onnx.export(
            model.model,
            dummy_input,
            str(onnx_path),
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            do_constant_folding=True
        )
        logging.info(f"Model exported to ONNX at {onnx_path}")
    except Exception as e:
        logging.error(f"ONNX export failed: {e}")
        sys.exit(1)
    return onnx_path

def upload_to_s3(local_path: str, bucket: str, s3_folder: str):
    s3_client = boto3.client('s3')
    if os.path.isfile(local_path):
        filename = os.path.basename(local_path)
        s3_path = f"{s3_folder}/{filename}"
        s3_client.upload_file(local_path, bucket, s3_path)
        logging.info(f"Uploaded {local_path} to s3://{bucket}/{s3_path}")
        return f"s3://{bucket}/{s3_path}"
    elif os.path.isdir(local_path):
        for root, dirs, files in os.walk(local_path):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, local_path)
                s3_path = f"{s3_folder}/{rel_path}"
                s3_client.upload_file(full_path, bucket, s3_path)
                logging.info(f"Uploaded {full_path} to s3://{bucket}/{s3_path}")
        return f"s3://{bucket}/{s3_folder}"
    else:
        raise ValueError(f"{local_path} is not a valid file or directory")

def parse_args():
    parser = argparse.ArgumentParser(description="Train, evaluate, export YOLO model to ONNX, upload to S3")
    parser.add_argument('--model', type=str, default="yolov10n.pt")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='auto')
    parser.add_argument('--initial_learning_rate', type=float, default=0.01)
    parser.add_argument('--final_learning_rate', type=float, default=0.01)
    return parser.parse_args()

def main():
    install_packages()
    args = parse_args()
    data_yaml = create_data_yaml(PROCESSING_INPUT_DIR, Path(PROCESSING_INPUT_DIR) / "data.yaml")
    model = train_yolo(args.model, str(data_yaml), args)
    evaluate_model(model, str(data_yaml), EVALUATION_OUTPUT_DIR)
    onnx_path = export_to_onnx(model, EVALUATION_OUTPUT_DIR)

    # Upload ONNX and metrics to S3
    upload_to_s3(EVALUATION_OUTPUT_DIR, S3_BUCKET, f"{S3_FOLDER}/evaluation")
    logging.info("All outputs uploaded to S3 successfully.")

if __name__ == "__main__":
    main()
