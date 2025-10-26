"""
Модуль для обучения модели
"""
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.config import DATA_ROOT

def train_one_epoch(model, optimizer, data_loader, device, epoch, stage="A"):
    """
    Одна эпоха обучения
    
    Args:
        model: модель для обучения
        optimizer: оптимизатор
        data_loader: DataLoader с тренировочными данными
        device: устройство (cuda/cpu)
        epoch (int): номер эпохи
        stage (str): этап обучения ('A' или 'B')
    """
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(
        data_loader, 
        desc=f"Stage {stage}, Epoch {epoch+1} [Train]", 
        leave=False
    )

    avg_losses_list = list()
    
    for images, targets in progress_bar:
        # Перемещаем данные на устройство
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Прямой проход и вычисление потерь
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Обратный проход
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Обновляем статистику
        total_loss += losses.item()
        avg_loss = total_loss / (progress_bar.n + 1)

        avg_losses_list.append(avg_loss)

        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
    try:
        loss_df = pd.DataFrame({"batch_num": range(len(avg_losses_list)),
                                "avg_loss": avg_losses_list})

        loss_dir_path = Path(f"{DATA_ROOT}/training_losses")
        loss_dir_path.mkdir(parents=True, exist_ok=True)

        loss_file_path = f"{loss_dir_path}/losses_stage_{stage}_epoch_{epoch+1}.csv"
        loss_df.to_csv(loss_file_path, index=False)
        
        print(f"Потери сохранены в {loss_file_path}")
    except Exception as e:
        print(f"Произошла ошибка при сохранении losses в {loss_dir_path}/losses_stage_{stage}_epoch_{epoch+1}.csv: {e}")
    return avg_loss

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """
    Сохраняет чекпоинт модели
    
    Args:
        model: модель
        optimizer: оптимизатор
        scheduler: планировщик learning rate
        epoch (int): номер эпохи
        loss (float): значение потерь
        filepath (str): путь для сохранения
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    torch.save(checkpoint, filepath)
    print(f"Чекпоинт сохранен: {filepath}")

def save_best_model(model, filepath):
    """
    Сохраняет только веса лучшей модели для инференса
    
    Args:
        model: модель
        filepath (str): путь для сохранения
    """
    torch.save(model.state_dict(), filepath)
    print(f"Лучшая модель сохранена: {filepath}")