"""
Вспомогательные функции для обучения
"""
import torch

def setup_optimizer(model, learning_rate, momentum=0.9, weight_decay=0.0005):
    """
    Настраивает оптимизатор для модели
    
    Args:
        model: модель для оптимизации
        learning_rate (float): скорость обучения
        momentum (float): момент
        weight_decay (float): вес decay
        
    Returns:
        torch.optim.Optimizer: оптимизатор
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=learning_rate, 
        momentum=momentum, 
        weight_decay=weight_decay
    )
    return optimizer

def setup_scheduler(optimizer, step_size=3, gamma=0.1):
    """
    Настраивает планировщик learning rate
    
    Args:
        optimizer: оптимизатор
        step_size (int): шаг уменьшения LR
        gamma (float): множитель для LR
        
    Returns:
        torch.optim.lr_scheduler: планировщик
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)