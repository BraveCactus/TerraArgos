"""
Функции для создания DataLoader
"""
from torch.utils.data import DataLoader
from src.config import BATCH_SIZE

def collate_fn(batch):
    """
    Collate function для батчирования данных детекции
    
    Args:
        batch: список элементов (image, target)
        
    Returns:
        tuple: (images, targets) где images и targets - списки
    """
    return tuple(zip(*batch))

def create_data_loaders(train_dataset, val_dataset, batch_size=BATCH_SIZE):
    """
    Создает DataLoader для тренировочного и валидационного наборов
    
    Args:
        train_dataset: тренировочный датасет
        val_dataset: валидационный датасет
        batch_size: размер батча
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader