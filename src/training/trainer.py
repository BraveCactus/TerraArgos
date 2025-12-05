"""
Модуль для обучения модели
"""
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.config import DATA_ROOT, RESULTS_ROOT

def train_one_epoch(model_name, model, optimizer, data_loader, device, epoch, stage="A", filepath=RESULTS_ROOT):
    """
    Одна эпоха обучения
    
    Args:
        model_name: название модели
        model: модель для обучения
        optimizer: оптимизатор
        data_loader: DataLoader с тренировочными данными
        device: устройство (cuda/cpu)
        epoch (int): номер эпохи
        stage (str): этап обучения ('A' или 'B')
        filepath: путь для сохранения (по умолчанию RESULTS_ROOT)
    
    Returns:
        float: средний loss за эпоху
    """ 
   
    model.train()
    
    batch_losses = []
    total_loss = 0.0
    batch_count = 0
    
    progress_bar = tqdm(
        data_loader, 
        desc=f"Stage {stage}, Epoch {epoch+1} [Train]", 
        leave=False
    )
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        try:
            # Перемещаем данные на устройство
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Прямой проход и вычисление потерь
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Обратный проход
            optimizer.zero_grad()
            losses.backward()            
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
           
            batch_loss = losses.item()
            batch_losses.append(batch_loss)
            total_loss += batch_loss
            batch_count += 1
            
            # Вычисляем текущее скользящее среднее для progress bar
            current_avg_loss = total_loss / batch_count
            progress_bar.set_postfix(loss=f"{current_avg_loss:.4f}")
            
        except RuntimeError as e:
            print(f"\nОшибка при обучении в батче {batch_idx+1}: {e}")       
    
    
    try:
        loss_dir_path = Path(filepath) / f"{model_name}_results" / "training_losses"
        loss_dir_path.mkdir(parents=True, exist_ok=True)        
       
        loss_df = pd.DataFrame({
            "batch_num": range(len(batch_losses)),
            "batch_loss": batch_losses,       
            "stage": stage,
            "epoch": epoch + 1
        })        

        avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
        summary_loss_df = pd.DataFrame({
            "stage": [stage],
            "epoch": [epoch + 1],
            "avg_loss": [avg_loss],
            "num_batches": [len(batch_losses)]
        })
       
        loss_file_path = loss_dir_path / f"losses_stage_{stage}_epoch_{epoch+1}.csv"
        loss_df.to_csv(loss_file_path, index=False)        
        
        all_losses_file = loss_dir_path / "all_losses.csv"       
        
        summary_loss_df.to_csv(all_losses_file, mode='a', header=not all_losses_file.exists(), index=False)
        
        print(f"Потери сохранены: {loss_file_path}")        
        
    except Exception as e:
        print(f"Ошибка при сохранении losses: {e}")    
    
    return sum(batch_losses) / len(batch_losses) if batch_losses else 0.0

def save_checkpoint(model_name, model, optimizer, scheduler, stage, epoch, loss, filepath=RESULTS_ROOT):
    """
    Сохраняет чекпоинт модели
    
    Args:
        model_name: название модели
        model: модель
        optimizer: оптимизатор
        scheduler: планировщик learning rate
        stage (str): этап обучения ('A' или 'B')
        epoch (int): номер эпохи
        loss (float): значение потерь
        filepath (str): путь для сохранения
    """
    checkpoint = {
        'stage': stage,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }

    checkpoint_dir = Path(filepath) / f"{model_name}_results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_checkpoint_stage_{stage}_epoch_{epoch+1}.pt"
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Чекпоинт сохранен: {checkpoint_path}")

def save_best_model(model_name, model, filepath=RESULTS_ROOT):
    """
    Сохраняет только веса лучшей модели для инференса
    
    Args:
        model_name: название модели
        model: модель
        filepath (str): путь для сохранения
    """
    best_model_dir = Path(filepath) / f"{model_name}_results" / "checkpoints" / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = Path(filepath) / f"{model_name}_results" / "metrics"
    metrics = pd.read_csv(metrics_dir / "all_summaries.csv")

    best_epoch_row = metrics.loc[metrics['f1'].idxmax()]
    stage = best_epoch_row['stage']
    epoch_num = best_epoch_row['epoch']

    best_f1 = best_epoch_row['f1']
    best_model_path = best_model_dir / f"{model_name}_best_stage_{stage}_epoch_{epoch_num}.pt"
    
    torch.save(model.state_dict(), best_model_path)
    print(f"Лучшая модель сохранена (F1={best_f1:.4f}, stage={stage}, epoch={epoch_num}): {best_model_path}")