# 🎨 Paint Defect Detection System

> AI-powered automotive paint surface quality inspection using YOLOv8 Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![Demo](docs/demo.gif)

## 📋 Overview

Advanced paint defect detection system for automotive quality control. Uses YOLOv8 segmentation to detect and classify surface defects including dirt, paint runs, scratches, and water marks.

### ✨ Features

- 🎯 **Real-time Detection** - Fast inference (~50ms per image)
- 🔍 **Instance Segmentation** - Precise defect localization with polygon masks
- 📊 **Comprehensive Analysis** - Detailed statistics and confidence scores
- 💾 **Export Results** - Save annotated images, JSON reports, and CSV data
- 🖥️ **Modern GUI** - Beautiful Tkinter-based desktop application
- 🔄 **Batch Processing** - Process multiple images at once
- 📈 **Training Pipeline** - Easy model training on custom datasets

### 🎭 Detected Defect Types

| Defect Type | Description | Icon |
|------------|-------------|------|
| Dirt | Surface contamination | 🟤 |
| Paint Runs | Paint dripping/sagging | 💧 |
| Scratches | Surface abrasion | 🔴 |
| Water Marks | Water spots/stains | 💦 |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Installation

```bash
# Clone repository
git clone https://github.com/Hacksutates/paint-defect-detection.git
cd paint-defect-detection

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Desktop Application (Recommended)

```bash
python app.py
```

**Features:**
- Drag & drop image upload
- Real-time detection preview
- Adjustable confidence threshold
- One-click result export

#### 2. Command Line Detection

```python
from detector import PaintDefectDetector

# Initialize detector
detector = PaintDefectDetector(
    model_path='model/best.pt',
    threshold=0.3
)

# Detect defects
result = detector.detect('path/to/image.jpg')

# Save results
detector.save_results(result, output_dir='results/')
```

#### 3. Batch Processing

```bash
python batch_process.py --input folder/with/images/ --output results/
```

---

## 🎓 Training Your Own Model

### 1. Prepare Dataset

Organize your dataset in YOLO segmentation format:

```
dataset22/
├── train/
│   ├── images/
│   └── labels/  # Polygon annotations
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

**Important:** Labels must be polygons (not bounding boxes). Each line format:
```
class_id x1 y1 x2 y2 x3 y3 ...
```

### 2. Verify Dataset

```bash
python check_dataset.py --dataset path/to/dataset22
```

### 3. Start Training

```bash
python train.py --dataset path/to/dataset22 --epochs 100 --batch 16
```

**Training Options:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--imgsz`: Image size (default: 640)
- `--model`: Model size: n/s/m/l/x (default: s)

### 4. Resume Training

If training was interrupted:

```bash
python continue.py --run allur_defects1 --epochs 50
```

---

## 📊 Project Structure

```
paint-defect-detection/
├── app.py                    # Desktop GUI application
├── detector.py               # Core detection module
├── trainer.py                # Training pipeline
├── batch_process.py          # Batch processing script
├── check_dataset.py          # Dataset validation
├── continue.py               # Resume training
├── convert.py                # Convert bbox to polygons
├── requirements.txt          # Python dependencies
├── model/
│   └── best.pt              # Trained model weights
├── dataset22/               # Training dataset
├── tests/                   # Test images
├── runs/                    # Training runs
└── docs/                    # Documentation
```

---

## 🎯 API Reference

### PaintDefectDetector

```python
class PaintDefectDetector:
    def __init__(self, model_path: str, threshold: float = 0.3)
    def detect(self, image_path: str) -> Dict
    def detect_batch(self, image_paths: List[str]) -> List[Dict]
    def visualize(self, image_path: str, save_path: str = None)
    def save_results(self, results: Dict, output_dir: str)
```

### Detection Result Format

```python
{
    'has_defect': bool,
    'num_defects': int,
    'defects': [
        {
            'class': str,           # Defect type
            'confidence': float,    # 0.0 - 1.0
            'area_percent': float,  # Defect area %
            'bbox': [x1, y1, x2, y2],
            'severity': str         # Low/Medium/High
        }
    ],
    'inference_time_ms': float,
    'image_size': (height, width)
}
```

---

## 🛠️ Configuration

### Model Settings

Edit `config.yaml`:

```yaml
model:
  type: yolov8s-seg
  confidence_threshold: 0.3
  iou_threshold: 0.45

classes:
  0: dirt
  1: runs
  2: scratch
  3: water marks

training:
  epochs: 100
  batch_size: 16
  image_size: 640
  patience: 50
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Inference Time | ~45ms (GPU) / ~200ms (CPU) |
| mAP50 (Box) | 0.852 |
| mAP50 (Mask) | 0.847 |
| Model Size | 25 MB |
| Input Resolution | 640×640 |

**Tested on:**
- GPU: NVIDIA GTX 1650
- CPU: Intel Core i5-9300H

---

## 📸 Screenshots

### Desktop Application
![App Screenshot](docs/screenshot_app.png)

### Detection Results
![Results](docs/screenshot_results.png)

### Training Metrics
![Training](docs/training_metrics.png)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 🐛 Troubleshooting

### Common Issues

**1. "Model not found"**
```bash
# Download pre-trained model or place your trained model:
cp path/to/best.pt model/best.pt
```

**2. "CUDA out of memory"**
```python
# Reduce batch size in training:
python train.py --batch 8  # or --batch 4
```

**3. "Labels are bounding boxes, not polygons"**
```bash
# Convert bbox to polygons:
python convert.py --dataset path/to/dataset22
```

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [Roboflow](https://roboflow.com) - Dataset management
- Dataset from [Roboflow Universe](https://universe.roboflow.com/poli-h7nww/final-year-car-paint-defect)

---

## 📧 Contact

**Author:** Alimzhan Zhangalishev

- Email: hacksutates@gmail.com
- GitHub: [@Hacksutates](https://github.com/Hacksutates)

---

<p align="center">
  Made with ❤️ for quality assurance in automotive industry
</p>
