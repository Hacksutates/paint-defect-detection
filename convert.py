"""
Конвертирует bbox в полигоны для сегментации
"""

from pathlib import Path

def convert_bbox_to_polygon(label_path: Path):
    """Конвертирует YOLO bbox в прямоугольный полигон"""
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    converted = []
    
    for line in lines:
        parts = line.strip().split()
        
        if len(parts) == 5:  # Это bbox: class x_center y_center width height
            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Конвертируем в 4-точечный полигон (прямоугольник)
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center - height / 2
            x3 = x_center + width / 2
            y3 = y_center + height / 2
            x4 = x_center - width / 2
            y4 = y_center + height / 2
            
            poly_line = f"{class_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n"
            converted.append(poly_line)
            
        else:  # Уже полигон
            converted.append(line)
    
    # Перезаписываем файл
    with open(label_path, 'w') as f:
        f.writelines(converted)


def convert_dataset(dataset_path: str):
    """Конвертирует весь датасет"""
    dataset_path = Path(dataset_path)
    
    print("🔄 Конвертация bbox → polygons...")
    
    for split in ['train', 'valid', 'test']:
        label_dir = dataset_path / split / 'labels'
        
        if not label_dir.exists():
            continue
        
        label_files = list(label_dir.glob('*.txt'))
        print(f"\n{split}: {len(label_files)} файлов")
        
        for label_file in label_files:
            convert_bbox_to_polygon(label_file)
        
        print(f"✅ {split} конвертирован")
    
    print("\n✅ ГОТОВО! Теперь можешь обучать с -seg моделью")


if __name__ == "__main__":
    # ВАЖНО: Сделай backup перед запуском!
    convert_dataset('dataset22')