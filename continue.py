"""
Продолжение обучения YOLOv8 с последнего чекпоинта
"""

from ultralytics import YOLO
from pathlib import Path

def resume_training(run_name='allur_defects12', additional_epochs=50):
    """
    Возобновление обучения с последнего чекпоинта
    
    Args:
        run_name: имя эксперимента (папка в runs/segment/)
        additional_epochs: сколько еще эпох обучать
    """
    
    # Путь к последнему чекпоинту
    last_checkpoint = Path(f'runs/segment/{run_name}/weights/last.pt')
    
    if not last_checkpoint.exists():
        print(f"❌ Чекпоинт не найден: {last_checkpoint}")
        print("\nДоступные эксперименты:")
        
        runs_dir = Path('runs/segment')
        if runs_dir.exists():
            for run_dir in sorted(runs_dir.iterdir()):
                if run_dir.is_dir():
                    last_pt = run_dir / 'weights' / 'last.pt'
                    if last_pt.exists():
                        print(f"  ✅ {run_dir.name}")
        return None
    
    print(f"📍 Найден чекпоинт: {last_checkpoint}")
    
    # Загрузка модели с чекпоинта
    model = YOLO(str(last_checkpoint))
    
    print(f"\n🔄 Продолжение обучения...")
    print(f"   Дополнительно эпох: {additional_epochs}")
    print(f"   Сохранение в: runs/segment/{run_name}\n")
    
    # ВАЖНО: используй resume=True
    results = model.train(
        resume=True,  # ← Это ключевой параметр!
        epochs=additional_epochs,  # Дополнительные эпохи
    )
    
    print("\n✅ Обучение завершено!")
    print(f"📁 Результаты: runs/segment/{run_name}/weights/")
    
    return results


def resume_from_specific_epoch(checkpoint_path, total_epochs=150):
    """
    Продолжить с конкретного чекпоинта до определенного количества эпох
    
    Args:
        checkpoint_path: путь к конкретному .pt файлу
        total_epochs: до какой эпохи обучать (не дополнительно, а всего)
    """
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Файл не найден: {checkpoint_path}")
        return None
    
    print(f"📍 Загрузка чекпоинта: {checkpoint_path}")
    
    model = YOLO(str(checkpoint_path))
    
    # Узнать на какой эпохе остановились
    # (это можно посмотреть в results.csv или args.yaml)
    
    print(f"\n🔄 Продолжение обучения до эпохи {total_epochs}")
    
    results = model.train(
        resume=True,
        epochs=total_epochs
    )
    
    print("\n✅ Готово!")
    
    return results


def continue_with_new_params(run_name='allur_defects12', 
                             epochs=50, 
                             new_batch_size=8,
                             new_lr=0.0005):
    """
    Продолжить обучение с НОВЫМИ параметрами
    (файнтюнинг с другими гиперпараметрами)
    """
    
    last_checkpoint = Path(f'runs/segment/{run_name}/weights/last.pt')
    
    if not last_checkpoint.exists():
        print(f"❌ Чекпоинт не найден: {last_checkpoint}")
        return None
    
    print(f"📍 Загрузка: {last_checkpoint}")
    print(f"⚙️ Новые параметры:")
    print(f"   Batch size: {new_batch_size}")
    print(f"   Learning rate: {new_lr}")
    
    model = YOLO(str(last_checkpoint))
    
    # Продолжение с новыми параметрами
    results = model.train(
        data='dataset22/data.yaml',  # Укажи свой датасет
        epochs=epochs,
        batch=new_batch_size,
        lr0=new_lr,
        project='runs/segment',
        name=f'{run_name}_continued',  # Новое имя чтобы не перезаписать
        exist_ok=False
    )
    
    print("\n✅ Обучение с новыми параметрами завершено!")
    
    return results


# ============================================================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ============================================================================

def example_1_simple_resume():
    """
    Пример 1: Простое продолжение обучения
    Автоматически продолжит с того места где остановился
    """
    print("="*60)
    print("📘 Пример 1: Простое возобновление")
    print("="*60)
    
    # Просто возобновляет с последнего чекпоинта
    resume_training(
        run_name='allur_defects12',  # Имя твоего эксперимента
        additional_epochs=50          # Еще 50 эпох
    )


def example_2_resume_to_target():
    """
    Пример 2: Продолжить до конкретной эпохи
    Например, остановился на 45, хочешь дойти до 100
    """
    print("="*60)
    print("📘 Пример 2: Продолжить до эпохи 100")
    print("="*60)
    
    resume_from_specific_epoch(
        checkpoint_path='runs/segment/allur_defects12/weights/last.pt',
        total_epochs=100  # Дойти до 100 эпох
    )


def example_3_finetune():
    """
    Пример 3: Файнтюнинг с другими параметрами
    Если хочешь поэкспериментировать с гиперпараметрами
    """
    print("="*60)
    print("📘 Пример 3: Файнтюнинг с новыми параметрами")
    print("="*60)
    
    continue_with_new_params(
        run_name='allur_defects12',
        epochs=30,
        new_batch_size=8,   # Уменьшил batch для экономии памяти
        new_lr=0.0001       # Меньший learning rate для стабильности
    )


def check_training_progress():
    """
    Проверить прогресс обучения - на какой эпохе остановился
    """
    import pandas as pd
    
    results_csv = Path('runs/segment/allur_defects1/results.csv')
    
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        last_epoch = df['epoch'].max()
        
        print(f"\n📊 Статус обучения:")
        print(f"   Последняя эпоха: {int(last_epoch)}")
        print(f"   Всего эпох пройдено: {len(df)}")
        
        # Последние метрики
        last_row = df.iloc[-1]
        print(f"\n📈 Последние метрики:")
        print(f"   Box mAP50: {last_row.get('metrics/mAP50(B)', 'N/A'):.3f}")
        print(f"   Mask mAP50: {last_row.get('metrics/mAP50(M)', 'N/A'):.3f}")
        
        return int(last_epoch)
    else:
        print(f"❌ Файл результатов не найден: {results_csv}")
        return None


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ YOLOV8                              ║
    ║   Resume Training Script                                   ║
    ╚════════════════════════════════════════════════════════════╝
    
    Доступные функции:
    
    1. check_training_progress()     - Проверить на какой эпохе остановился
    2. example_1_simple_resume()     - Просто продолжить обучение
    3. example_2_resume_to_target()  - Продолжить до конкретной эпохи
    4. example_3_finetune()          - Продолжить с новыми параметрами
    
    """)
    
    # Сначала проверь прогресс
    print("🔍 Проверка прогресса обучения...\n")
    check_training_progress()
    
    print("\n" + "="*60)
    print("Выбери пример для запуска:")
    print("="*60)
    
    # Раскомментируй нужный пример:
    
    # example_1_simple_resume()
    example_2_resume_to_target()
    # example_3_finetune()