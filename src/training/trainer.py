"""
Модуль для обучения модели
"""
import torch
from tqdm import tqdm
from pathlib import Path

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
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
    
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