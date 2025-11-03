"""
Улучшенный скрипт обучения с CLI интерфейсом
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

def validate_dataset(dataset_path):
    """Быстрая проверка датасета"""
    dataset_path = Path(dataset_path)
    
    # Проверка data.yaml
    data_yaml = dataset_path / 'data.yaml'
    if not data_yaml.exists():
        print(f"❌ Error: data.yaml not found in {dataset_path}")
        return False
    
    # Проверка папок
    required_folders = ['train/images', 'train/labels', 'valid/images', 'valid/labels']
    for folder in required_folders:
        folder_path = dataset_path / folder
        if not folder_path.exists():
            print(f"❌ Error: Missing folder {folder}")
            return False
    
    print(f"✅ Dataset structure is valid")
    return True


def train_model(args):
    """Обучение модели"""
    
    print("\n" + "="*70)
    print("🎨 PAINT DEFECT DETECTION - TRAINING")
    print("="*70 + "\n")
    
    # Валидация датасета
    print("📋 Step 1/3: Validating dataset...")
    if not validate_dataset(args.dataset):
        print("\n❌ Dataset validation failed. Please check your dataset structure.")
        sys.exit(1)
    
    # Загрузка модели
    print(f"\n📦 Step 2/3: Loading YOLOv8{args.model}-seg model...")
    model = YOLO(f'yolov8{args.model}-seg.pt')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    
    # Подготовка параметров
    dataset_path = Path(args.dataset).resolve()
    data_yaml = dataset_path / 'data.yaml'
    
    print(f"\n🚀 Step 3/3: Starting training...")
    print(f"   Dataset: {dataset_path}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Output: runs/segment/{args.name}\n")
    
    # Обучение
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            workers=args.workers,
            project='runs/segment',
            name=args.name,
            exist_ok=args.exist_ok,
            device=device,
            
            # Оптимизации
            amp=True,
            cos_lr=True,
            close_mosaic=10,
            plots=True,
            save=True,
            save_period=args.save_period
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📁 Model saved to: runs/segment/{args.name}/weights/")
        print(f"   • best.pt  - Best model (highest mAP)")
        print(f"   • last.pt  - Latest checkpoint")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 Segmentation model for paint defect detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train.py --dataset dataset22 --epochs 100
  
  # Custom configuration
  python train.py --dataset dataset22 --epochs 50 --batch 8 --model n
  
  # With specific name
  python train.py --dataset dataset22 --name my_experiment --epochs 100
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset folder containing data.yaml'
    )
    
    # Optional arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16). Reduce if GPU memory is low'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training (default: 640)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['n', 's', 'm', 'l', 'x'],
        default='s',
        help='Model size: n(nano), s(small), m(medium), l(large), x(xlarge) (default: s)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='allur_defects',
        help='Experiment name (default: allur_defects)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience in epochs (default: 50)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of dataloader workers (default: 8)'
    )
    
    parser.add_argument(
        '--save-period',
        type=int,
        default=-1,
        help='Save checkpoint every N epochs (-1 to disable) (default: -1)'
    )
    
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='Overwrite existing experiment with same name'
    )
    
    args = parser.parse_args()
    
    # Запуск обучения
    train_model(args)


if __name__ == "__main__":
    main()