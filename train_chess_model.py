"""
Script to train a YOLO11 model for chess piece detection.

Note: You need a chess piece dataset in YOLO format with the following structure:
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml

Chess piece classes (8 total):
0: white_pawn
1: white_knight
2: white_bishop
3: white_rook
4: white_queen
5: white_king
6: black_pawn
7: black_knight
8: black_bishop
9: black_rook
10: black_queen
11: black_king
"""

from ultralytics import YOLO
import argparse


def train_chess_model(dataset_yaml, epochs=100, imgsz=640, batch_size=16, device=0):
    """
    Train a YOLO11 model for chess piece detection.
    
    Args:
        dataset_yaml: Path to dataset.yaml file
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size for training
        device: GPU device ID (0 for first GPU, -1 for CPU)
    """
    print("[INFO] Loading YOLO11 model...")
    model = YOLO('yolov8n.yaml')  # nano model for faster training
    
    print(f"[INFO] Starting training with {epochs} epochs...")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {'GPU' if device >= 0 else 'CPU'}")
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        patience=20,  # Early stopping patience
        save=True,
        project='chess_models',
        name='chess_pieces',
        verbose=True
    )
    
    print("\n[✅] Training completed!")
    print(f"[INFO] Model saved to: chess_models/chess_pieces/weights/best.pt")
    
    return model


def validate_model(model_path, dataset_yaml):
    """
    Validate a trained model.
    
    Args:
        model_path: Path to trained model weights
        dataset_yaml: Path to dataset.yaml
    """
    print(f"[INFO] Validating model: {model_path}")
    
    model = YOLO(model_path)
    metrics = model.val(data=dataset_yaml)
    
    print("\n[INFO] Validation Results:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 for chess piece detection')
    parser.add_argument('--dataset', type=str, default='chess_dataset/data.yaml',
                       help='Path to dataset.yaml file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID (0 for first GPU, -1 for CPU)')
    parser.add_argument('--validate', type=str, default=None,
                       help='Validate existing model (provide path to weights)')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_model(args.validate, args.dataset)
    else:
        train_chess_model(
            dataset_yaml=args.dataset,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            device=args.device
        )


if __name__ == '__main__':
    main()
