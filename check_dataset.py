"""
Скрипт полной проверки датасета перед обучением
"""

from pathlib import Path
import yaml

def validate_allur_dataset(dataset_path='dataset22'):
    """Полная проверка датасета"""
    dataset_path = Path(dataset_path)
    
    print("="*70)
    print("🔍 ПРОВЕРКА ДАТАСЕТА ALLUR")
    print("="*70)
    
    issues = []
    
    # 1. Проверка data.yaml
    print("\n📄 Проверка data.yaml...")
    data_yaml = dataset_path / 'data.yaml'
    
    if not data_yaml.exists():
        issues.append("❌ КРИТИЧНО: Отсутствует data.yaml")
        print("❌ data.yaml НЕ НАЙДЕН!")
    else:
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        print(f"✅ data.yaml найден")
        print(f"   Путь к датасету: {data.get('path', 'НЕ УКАЗАН')}")
        print(f"   Train: {data.get('train', 'НЕ УКАЗАН')}")
        print(f"   Val: {data.get('val', 'НЕ УКАЗАН')}")
        print(f"   Test: {data.get('test', 'НЕ УКАЗАН')}")
        
        # Проверка классов
        names = data.get('names', [])
        nc = data.get('nc', 0)
        
        if isinstance(names, dict):
            print(f"   Классы ({nc}): {list(names.values())}")
        elif isinstance(names, list):
            print(f"   Классы ({nc}): {names}")
        else:
            issues.append("⚠️ Формат 'names' неясен")
        
        # Проверка путей
        if data.get('path') not in ['.', 'dataset22', str(dataset_path)]:
            issues.append(f"⚠️ path в data.yaml: '{data.get('path')}' - может быть некорректным")
    
    # 2. Проверка папок и файлов
    print("\n📁 Проверка структуры папок...")
    
    splits = {
        'train': {'images': 0, 'labels': 0},
        'valid': {'images': 0, 'labels': 0},
        'test': {'images': 0, 'labels': 0}
    }
    
    for split in ['train', 'valid', 'test']:
        img_dir = dataset_path / split / 'images'
        lbl_dir = dataset_path / split / 'labels'
        
        if img_dir.exists():
            images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            splits[split]['images'] = len(images)
            print(f"✅ {split}/images: {len(images)} файлов")
        else:
            issues.append(f"❌ Отсутствует папка: {split}/images")
        
        if lbl_dir.exists():
            labels = list(lbl_dir.glob('*.txt'))
            splits[split]['labels'] = len(labels)
            print(f"✅ {split}/labels: {len(labels)} файлов")
        else:
            issues.append(f"❌ Отсутствует папка: {split}/labels")
        
        # Проверка соответствия
        if splits[split]['images'] != splits[split]['labels']:
            issues.append(
                f"⚠️ {split}: images ({splits[split]['images']}) ≠ "
                f"labels ({splits[split]['labels']})"
            )
    
    # 3. Проверка формата аннотаций (СЕГМЕНТАЦИЯ!)
    print("\n🏷️ Проверка формата labels (полигоны для сегментации)...")
    
    sample_label_path = dataset_path / 'train' / 'labels'
    sample_files = list(sample_label_path.glob('*.txt'))[:3]
    
    if sample_files:
        for sample in sample_files:
            with open(sample, 'r') as f:
                lines = f.readlines()
            
            if lines:
                first_line = lines[0].strip()
                coords = first_line.split()
                
                # class_id x1 y1 x2 y2 x3 y3 ... (минимум 8 координат = 4 точки)
                num_coords = len(coords) - 1  # Вычитаем class_id
                
                if num_coords < 6:
                    issues.append(
                        f"❌ КРИТИЧНО: {sample.name} содержит bbox вместо полигона! "
                        f"({num_coords} координат, нужно минимум 6)"
                    )
                    print(f"❌ {sample.name}: BBOX формат (всего {len(coords)} значений)")
                    print(f"   Первая строка: {first_line[:50]}...")
                else:
                    print(f"✅ {sample.name}: Полигон ({num_coords} координат)")
    else:
        issues.append("⚠️ Нет файлов labels для проверки")
    
    # 4. Итоговый отчет
    print("\n" + "="*70)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("="*70)
    
    if issues:
        print(f"\n⚠️ ОБНАРУЖЕНО ПРОБЛЕМ: {len(issues)}\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        print("\n❌ ИСПРАВЬ ПРОБЛЕМЫ ПЕРЕД ОБУЧЕНИЕМ!\n")
        return False
    else:
        print("\n✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        print(f"\n📈 Статистика:")
        print(f"   Train:  {splits['train']['images']} изображений")
        print(f"   Valid:  {splits['valid']['images']} изображений")
        print(f"   Test:   {splits['test']['images']} изображений")
        print(f"   ИТОГО:  {sum(s['images'] for s in splits.values())} изображений")
        print("\n🚀 ГОТОВ К ОБУЧЕНИЮ!\n")
        return True


if __name__ == "__main__":
    validate_allur_dataset('dataset22')