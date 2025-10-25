"""
Скрипт для обучения с уже существующими данными
"""
import sys
import os
import torch
from torch.utils.data import Subset
from pathlib import Path

# Добавляем src в путь для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import *
from src.data.dataset import CocoDetectionForFasterRCNN, get_num_classes, get_names_classes
from src.data.dataloader import create_data_loaders
from src.models.faster_rcnn import get_model, freeze_backbone, unfreeze_backbone
from src.models.metrics import calculate_metrics
from src.training.trainer import train_one_epoch, save_checkpoint, save_best_model
from src.training.utils import setup_optimizer, setup_scheduler

def check_data_exists():
    """Проверяет существование данных"""
    img_dir, train_ann, val_ann = get_data_paths()
    
    print("Проверка данных...")
    print(f"Папка с изображениями: {img_dir} - {'СУЩЕСТВУЕТ' if img_dir.exists() else 'НЕ СУЩЕСТВУЕТ'}")
    print(f"Тренировочные аннотации: {train_ann} - {'СУЩЕСТВУЕТ' if train_ann.exists() else 'НЕ СУЩЕСТВУЕТ'}")
    print(f"Валидационные аннотации: {val_ann} - {'СУЩЕСТВУЕТ' if val_ann.exists() else 'НЕ СУЩЕСТВУЕТ'}")
    
    if not all([img_dir.exists(), train_ann.exists(), val_ann.exists()]):
        print("\nОШИБКА: Не все файлы данных найдены!")
        print("Убедитесь, что структура данных следующая:")
        print("VME_data/VME_CDSI_datasets/satellite_images/ [файлы .tif]")
        print("VME_data/VME_CDSI_datasets/annotations_HBB/train.json")
        print("VME_data/VME_CDSI_datasets/annotations_HBB/val.json")
        return False
    
    # Проверяем есть ли изображения в папке
    image_files = list(img_dir.glob("*.tif")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"Найдено файлов изображений: {len(image_files)}")
    
    if len(image_files) == 0:
        print("ОШИБКА: В папке с изображениями нет файлов!")
        return False
    
    return True

def main():
    print("=== Обучение Faster R-CNN с существующими данными ===")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"DEVICE: {DEVICE}")
    
    # 1. Проверка данных
    print("\n1. Проверка данных...")
    if not check_data_exists():
        return
    
    # Получаем пути к данным
    img_dir, train_ann, val_ann = get_data_paths()  
 
    # 2. Создание датасетов
    print("\n2. Создание датасетов...")
    try:
        train_dataset = CocoDetectionForFasterRCNN(root=str(img_dir), annFile=str(train_ann))
        val_dataset = CocoDetectionForFasterRCNN(root=str(img_dir), annFile=str(val_ann))
        
        # ОГРАНИЧИВАЕМ КОЛИЧЕСТВО ДАННЫХ ЕСЛИ ВКЛЮЧЕН DEBUG_MODE
        if DEBUG_MODE:
            train_dataset = Subset(train_dataset, indices=range(min(MAX_SAMPLES, len(train_dataset))))
            val_dataset = Subset(val_dataset, indices=range(min(MAX_SAMPLES, len(val_dataset))))

        print(f"Тренировочных samples: {len(train_dataset)}")
        print(f"Валидационных samples: {len(val_dataset)}")
        
        # Покажем пример данных
        sample_img, sample_target = train_dataset[0]
        print(f"Размер изображения: {sample_img.shape}")
        print(f"Количество объектов в примере: {len(sample_target['boxes'])}")
        
    except Exception as e:
        print(f"Ошибка при создании датасетов: {e}")
        return
    
    # 3. Создание модели
    print("\n3. Создание модели...")
    try:
        num_classes = get_num_classes(train_ann)
        model = get_model(num_classes)
        model.to(DEVICE)

        classes = get_names_classes(train_ann)

        print(f"Категорий в датасете: {len(classes)}")
        for name, id in classes.items():
            print(f"\t - {name}: (id:{id})")
            
    except Exception as e:
        print(f"Ошибка при создании модели: {e}")
        return
    
    # 4. Создание DataLoader
    print("\n4. Создание DataLoader...")
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)
    
    # 5. Настройка оптимизатора и планировщика
    print("\n5. Настройка оптимизатора...")
    optimizer = setup_optimizer(model, LEARNING_RATE_STAGE_A)
    scheduler = setup_scheduler(optimizer, STEP_SIZE, GAMMA)
    
    # Создаем папку для сохранения моделей
    model_dir = DATA_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    
    # 6. Стадия A: Замороженный backbone
    print("\n6. Стадия A: Обучение с замороженным backbone...")
    freeze_backbone(model)
    
    for epoch in range(NUM_EPOCHS_STAGE_A):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS_STAGE_A} ---")
        try:
            avg_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, "A")
            scheduler.step()

            # Вычисление метрик
            accuracy = calculate_metrics(model, val_loader, DEVICE)
            
            print(f"Результат эпохи {epoch+1}:")
            print(f"\tLoss: {avg_loss}")
            print(f"\Accuracy: {accuracy}")

            # Сохраняем чекпоинт
            checkpoint_path = model_dir / f"stage_a_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, checkpoint_path)
        except Exception as e:
            print(f"Ошибка при обучении эпохи {epoch+1}: {e}")
            break
    
    # 7. Стадия B: Размороженный backbone
    print("\n7. Стадия B: Обучение с размороженным backbone...")
    unfreeze_backbone(model)
    
    # Обновляем оптимизатор с меньшим LR
    optimizer = setup_optimizer(model, LEARNING_RATE_STAGE_B)
    scheduler = setup_scheduler(optimizer, STEP_SIZE, GAMMA)
    
    for epoch in range(NUM_EPOCHS_STAGE_B):
        print(f"\n--- Stage B, Epoch {epoch+1}/{NUM_EPOCHS_STAGE_B} ---")
        try:
            avg_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, "B")
            scheduler.step()
            
            # Сохраняем чекпоинт
            checkpoint_path = model_dir / f"stage_b_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, checkpoint_path)
        except Exception as e:
            print(f"Ошибка при обучении эпохи {epoch+1}: {e}")
            break
    
    # 8. Сохранение финальной модели
    print("\n8. Сохранение финальной модели...")
    final_model_path = model_dir / "final_faster_rcnn.pth"
    save_best_model(model, final_model_path)
    
    print("\n=== Обучение завершено! ===")
    print(f"Финальная модель сохранена: {final_model_path}")


if __name__ == "__main__":    
    main()