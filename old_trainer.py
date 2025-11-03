"""
Paint Defect Segmentation System - Для датасета Allur
Обнаружение дефектов покраски с сегментацией полигонов
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import time
import json

class PaintDefectSegmentationDetector:
    """
    Система обнаружения дефектов покраски с сегментацией
    Поддержка YOLO Segmentation формата (polygons)
    """
    
    def __init__(self, model_path: str = 'yolov8n-seg.pt', confidence_threshold: float = 0.3):
        """
        Инициализация детектора с сегментацией
        
        Args:
            model_path: путь к модели YOLOv8-seg (важно: -seg!)
            confidence_threshold: порог уверенности
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Устройство: {self.device}")
        
        # Загрузка YOLOv8 SEGMENTATION модели
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold
        
        # Названия классов дефектов (обнови под свой data.yaml)
        self.class_names = {
            0: 'dirt',
            1: 'runs',
            2: 'scratch',
            3: 'water marks'
        }

        
        print(f"✅ Модель загружена: {model_path}")
        print(f"📊 Порог: {confidence_threshold}")
        print(f"🏷️ Классы: {list(self.class_names.values())}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Предобработка изображения для металлических окрашенных поверхностей
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Улучшение для металлических поверхностей
        # 1. CLAHE для выделения дефектов
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def detect_defects(self, image_path: str, use_enhancement: bool = True) -> Dict:
        """
        Обнаружение дефектов с сегментацией
        
        Returns:
            {
                'has_defect': bool,
                'num_defects': int,
                'defects': [
                    {
                        'class': str,
                        'class_id': int,
                        'confidence': float,
                        'bbox': [x1, y1, x2, y2],
                        'mask': np.ndarray,  # Сегментационная маска
                        'area': float
                    }
                ],
                'inference_time_ms': float,
                'image_size': tuple
            }
        """
        start_time = time.time()
        
        # Загрузка и предобработка
        if use_enhancement:
            img = self.preprocess_image(image_path)
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Детекция с сегментацией
        results = self.model(img, conf=self.confidence_threshold, verbose=False)
        
        # Парсинг результатов
        has_defect = False
        defects_list = []
        
        if len(results) > 0:
            result = results[0]
            
            # Проверка наличия масок сегментации
            if result.masks is not None and len(result.masks) > 0:
                has_defect = True
                
                masks = result.masks.data.cpu().numpy()  # Маски сегментации
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(masks)):
                    mask = masks[i]
                    bbox = boxes[i]
                    conf = float(confidences[i])
                    cls_id = int(class_ids[i])
                    
                    # Расчет площади дефекта
                    area = np.sum(mask > 0.5) / (mask.shape[0] * mask.shape[1])
                    
                    defect_info = {
                        'class': self.class_names.get(cls_id, f'class_{cls_id}'),
                        'class_id': cls_id,
                        'confidence': conf,
                        'bbox': bbox.tolist(),
                        'mask': mask,
                        'area_percent': area * 100
                    }
                    
                    defects_list.append(defect_info)
        
        inference_time = (time.time() - start_time) * 1000
        
        result_dict = {
            'has_defect': has_defect,
            'num_defects': len(defects_list),
            'defects': defects_list,
            'inference_time_ms': inference_time,
            'image_size': img.shape[:2],
            'device': self.device
        }
        
        return result_dict
    
    def visualize_segmentation(self, image_path: str, save_path: str = None, 
                              show_masks: bool = True, show_boxes: bool = True):
        """
        Визуализация результатов сегментации
        """
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Детекция
        results = self.model(img_rgb, conf=self.confidence_threshold, verbose=False)
        
        # Создание визуализации
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Оригинал
        axes[0].imshow(img_rgb)
        axes[0].set_title('Оригинальное изображение')
        axes[0].axis('off')
        
        # С детекцией
        if len(results) > 0 and results[0].masks is not None:
            # Отрисовка масок и bbox
            annotated = results[0].plot(
                conf=True,
                labels=True,
                boxes=show_boxes,
                masks=show_masks
            )
            
            axes[1].imshow(annotated)
            
            # Статус
            if hasattr(results[0], 'masks') and results[0].masks is not None and hasattr(results[0].masks, 'data'):
                num_defects = results[0].masks.data.shape[0]
            else:
                num_defects = 0
            axes[1].set_title(f'❌ Обнаружено дефектов: {num_defects}', 
                            color='red', fontsize=14, weight='bold')
        else:
            axes[1].imshow(img_rgb)
            axes[1].set_title('✅ Дефектов не обнаружено', 
                            color='green', fontsize=14, weight='bold')
        
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"💾 Сохранено: {save_path}")
        
        plt.show()
    
    def detect_batch(self, image_paths: List[str], save_results: bool = True) -> List[Dict]:
        """
        Пакетная обработка с сохранением результатов
        """
        results = []
        
        print(f"\n🔄 Обработка {len(image_paths)} изображений...\n")
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] {Path(img_path).name}")
            
            try:
                result = self.detect_defects(img_path)
                results.append(result)
                
                # Вывод информации
                if result['has_defect']:
                    print(f"  ❌ ДЕФЕКТОВ: {result['num_defects']}")
                    for defect in result['defects']:
                        print(f"     • {defect['class']}: {defect['confidence']:.2%} "
                              f"(площадь: {defect['area_percent']:.2f}%)")
                else:
                    print(f"  ✅ Норма")
                
                print(f"  ⏱️ {result['inference_time_ms']:.1f}ms\n")
                
            except Exception as e:
                print(f"  ⚠️ Ошибка: {e}\n")
                results.append({'error': str(e), 'image_path': img_path})
        
        # Сохранение результатов
        if save_results:
            self.save_results_to_json(results, 'detection_results.json')
        
        return results
    
    def save_results_to_json(self, results: List[Dict], output_path: str):
        """
        Сохранение результатов в JSON (без масок - они большие)
        """
        # Убираем маски перед сохранением (они numpy arrays)
        results_clean = []
        for r in results:
            r_copy = r.copy()
            if 'defects' in r_copy:
                r_copy['defects'] = [
                    {k: v for k, v in d.items() if k != 'mask'}
                    for d in r_copy['defects']
                ]
            results_clean.append(r_copy)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Результаты сохранены: {output_path}")
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        Статистика по типам дефектов
        """
        total = len(results)
        defects_found = sum(1 for r in results if r.get('has_defect', False))
        
        # Статистика по классам
        class_counts = {}
        total_defects = 0
        
        for r in results:
            if 'defects' in r:
                for defect in r['defects']:
                    cls = defect['class']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                    total_defects += 1
        
        avg_time = np.mean([r.get('inference_time_ms', 0) for r in results])
        
        return {
            'total_images': total,
            'images_with_defects': defects_found,
            'defect_rate': defects_found / total if total > 0 else 0,
            'total_defects_found': total_defects,
            'defects_by_class': class_counts,
            'avg_inference_time_ms': avg_time
        }
    
    def print_statistics(self, stats: Dict):
        """Красивый вывод статистики"""
        print("\n" + "="*70)
        print("📊 СТАТИСТИКА ОБРАБОТКИ")
        print("="*70)
        print(f"Всего изображений:           {stats['total_images']}")
        print(f"С дефектами:                 {stats['images_with_defects']} "
              f"({stats['defect_rate']:.1%})")
        print(f"Всего дефектов обнаружено:   {stats['total_defects_found']}")
        print(f"\nПо типам дефектов:")
        for cls, count in stats['defects_by_class'].items():
            print(f"  • {cls:15s}: {count:3d}")
        print(f"\nСреднее время обработки:     {stats['avg_inference_time_ms']:.1f} ms")
        print("="*70)


# ============================================================================
# ТРЕНИРОВКА НА ТВОЕМ ДАТАСЕТЕ
# ============================================================================

class AllurDefectTrainer:
    """
    Специализированный тренер для датасета Allur
    """
    
    def __init__(self, dataset_path: str, model_size: str = 's'):
        """
        Args:
            dataset_path: путь к датасету (должен содержать data.yaml)
            model_size: 'n', 's', 'm', 'l', 'x'
        """
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        
        # ИСПРАВЛЕНО: добавлен -seg для segmentation модели!
        self.model = YOLO(f'yolov8{model_size}-seg.pt')
        
        print(f"🤖 Загружена YOLOv8{model_size}-seg")
    
    def verify_dataset_structure(self):
        """Улучшенная проверка структуры датасета"""
        print("\n🔍 Проверка датасета...")
        
        data_yaml = self.dataset_path / 'data.yaml'
        if not data_yaml.exists():
            print(f"❌ Не найден data.yaml в {self.dataset_path}")
            return False
        
        # Проверка обязательных папок
        required = ['train/images', 'valid/images', 'train/labels', 'valid/labels']
        for folder in required:
            path = self.dataset_path / folder
            if not path.exists():
                print(f"❌ Не найдена папка: {folder}")
                return False
            
            files = list(path.glob('*'))
            print(f"✅ {folder:20s}: {len(files)} файлов")
        
        # Проверка формата labels (должны быть полигоны для сегментации)
        sample_label = list((self.dataset_path / 'train/labels').glob('*.txt'))
        if sample_label:
            with open(sample_label[0], 'r') as f:
                first_line = f.readline().strip()
                coords = first_line.split()
                if len(coords) < 8:  # Минимум 4 точки (8 координат) для полигона
                    print("⚠️ ВНИМАНИЕ: labels могут содержать bbox вместо полигонов!")
                    print("   Для сегментации нужны полигоны (>6 координат)")
                else:
                    print(f"✅ Формат labels корректен (полигоны): {len(coords)} координат")
        
        print("✅ Структура датасета корректна!\n")
        return True

    def train(self, epochs: int = 100, imgsz: int = 640, batch: int = 16,
              patience: int = 50, workers: int = 8):
        """
        Обучение модели сегментации
        """
        if not self.verify_dataset_structure():
            print("❌ Исправь структуру датасета перед обучением")
            return None
        
        print(f"\n🚀 НАЧАЛО ОБУЧЕНИЯ")
        print(f"   Модель: YOLOv8{self.model_size}-seg")
        print(f"   Эпохи: {epochs}")
        print(f"   Размер изображений: {imgsz}")
        print(f"   Batch size: {batch}")
        print(f"   Устройство: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
        
        try:
            results = self.model.train(
                data=str(self.dataset_path / 'data.yaml'),  # Явное преобразование в str
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,
                workers=workers,
                save=True,
                project='runs/segment',
                name='allur_defects',
                plots=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                
                # Оптимизации для сегментации
                amp=True,  # Automatic Mixed Precision
                cos_lr=True,  # Cosine LR scheduler
                close_mosaic=10,  # Отключить mosaic за 10 эпох до конца
            )
            
            print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            print(f"📁 Модель сохранена: runs/segment/allur_defects/weights/best.pt")
            
            return results
            
        except Exception as e:
            print(f"\n❌ ОШИБКА ПРИ ОБУЧЕНИИ: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def validate(self, weights_path: str = None):
        """
        ИСПРАВЛЕННАЯ валидация обученной модели
        """
        if weights_path:
            model = YOLO(weights_path)
        else:
            model = self.model
        
        print("\n📊 Валидация модели...")
        
        try:
            metrics = model.val()
            
            print("\n" + "="*60)
            print("📊 МЕТРИКИ ВАЛИДАЦИИ:")
            print("="*60)
            
            # ИСПРАВЛЕНО: правильный доступ к метрикам через атрибуты
            if hasattr(metrics, 'box') and metrics.box is not None:
                print(f"Box mAP50:        {metrics.box.map50:.3f}")
                print(f"Box mAP50-95:     {metrics.box.map:.3f}")
            
            if hasattr(metrics, 'seg') and metrics.seg is not None:
                print(f"Mask mAP50:       {metrics.seg.map50:.3f}")
                print(f"Mask mAP50-95:    {metrics.seg.map:.3f}")
            
            # Если не удалось получить через атрибуты, пробуем другие способы
            if not hasattr(metrics, 'box') and not hasattr(metrics, 'seg'):
                print("⚠️ Стандартные атрибуты недоступны")
                print(f"Тип объекта metrics: {type(metrics)}")
                print(f"Доступные атрибуты: {[a for a in dir(metrics) if not a.startswith('_')]}")
                
                # Попытка прямого вывода
                print(f"\nПолный вывод metrics:\n{metrics}")
            
            print("="*60)
            
            return metrics
            
        except Exception as e:
            print(f"\n❌ ОШИБКА ПРИ ВАЛИДАЦИИ: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ============================================================================

def example_single_detection():
    """Пример: детекция на одном изображении"""
    detector = PaintDefectSegmentationDetector(
        model_path='runs/segment/allur_defects2/weights/best.pt',  # Сначала используем претрейн
        confidence_threshold=0.25
    )
    
    # Твое изображение
    image_path = 'DSC_0016_JPG.rf.64e4b9106da9afe5a72407ec53b4f59d.jpg'
    
    # Детекция
    result = detector.detect_defects(image_path)
    
    print(f"\n{'='*60}")
    if result['has_defect']:
        print(f"❌ ОБНАРУЖЕНО ДЕФЕКТОВ: {result['num_defects']}")
        for i, defect in enumerate(result['defects'], 1):
            print(f"\nДефект #{i}:")
            print(f"  Тип:         {defect['class']}")
            print(f"  Уверенность: {defect['confidence']:.2%}")
            print(f"  Площадь:     {defect['area_percent']:.2f}%")
    else:
        print("✅ Дефектов не обнаружено")
    
    print(f"\n⏱️ Время: {result['inference_time_ms']:.1f} ms")
    print(f"{'='*60}")
    
    # Визуализация
    detector.visualize_segmentation(image_path, save_path='result_seg.jpg')


def example_train_on_allur_dataset():
    """Пример: обучение на датасете Allur"""
    
    # ВАЖНО: Укажи путь к твоему датасету
    trainer = AllurDefectTrainer(
        dataset_path='dataset22',  # ИЗМЕНИ!
        model_size='s'  # small - баланс скорость/точность
    )
    
    # Обучение
    trainer.train(
        epochs=100,
        imgsz=640,
        batch=16,  # Уменьши если мало памяти GPU
        patience=50
    )
    
    # Валидация
    trainer.validate()


def example_use_trained_model():
    """Пример: использование обученной модели"""
    detector = PaintDefectSegmentationDetector(
        model_path='runs/segment/allur_defects/weights/best.pt',
        confidence_threshold=0.3
    )
    
    # Пакетная обработка тестовых изображений
    test_images = [
        'test1.jpg',
        'test2.jpg',
        'test3.jpg'
    ]
    
    results = detector.detect_batch(test_images)
    
    # Статистика
    stats = detector.get_statistics(results)
    detector.print_statistics(stats)


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║   СИСТЕМА СЕГМЕНТАЦИИ ДЕФЕКТОВ ПОКРАСКИ - Allur Dataset       ║
    ║   Paint Defect Segmentation System v2.0                        ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Возможности:
    ✓ Сегментация дефектов (маски полигонов)
    ✓ Поддержка классов: пропуски, пятна, царапины, затеки, избыток
    ✓ Точная локализация дефектов
    ✓ Расчет площади дефектов
    ✓ Обучение на твоем датасете Allur
    
    Примеры:
      example_single_detection()     - тест на одном изображении
      example_train_on_allur_dataset() - обучение на твоих данных
      example_use_trained_model()    - использование обученной модели
    """)
    
    # Раскомментируй:
    # example_single_detection()
    example_train_on_allur_dataset()
