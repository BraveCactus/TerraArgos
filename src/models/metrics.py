"""
Модуль с 3 метриками и сохранением истории
"""
from pathlib import Path
import torch
import pandas as pd
from torchvision.ops import box_iou
from src.config import BATCH_SIZE, DETECT_THRESHOLD, IOU_TRESHHOLD, DATA_ROOT
from src.visualization.metrics_plots import plot_metric_per_epoch, plot_all_metrics_comparison

# История для всех метрик
metrics_history = {
    'simple_count': [],      # Простая метрика счета
    'iou_basic': [],         # Базовая IoU метрика  
    'iou_advanced': []       # Продвинутая IoU метрика
}

def calculate_simple_count_metric(model, dataloader, device, stage, epoch):
    """
    МЕТРИКА 1: Simple Count - сравнение количества объектов
    """
    model.eval()    
    total_diff = 0
    total_true = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]            
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                above_threshold = pred['scores'] > DETECT_THRESHOLD
                pred_count = len(pred['boxes'][above_threshold])
                true_count = len(target['boxes'])
                
                total_diff += abs(pred_count - true_count)
                total_true += true_count

    # Accuracy = 1 - средняя ошибка в количестве
    if total_true > 0:
        accuracy = 1.0 - (total_diff / total_true)
        accuracy = max(0.0, min(1.0, accuracy))  # Ограничиваем от 0 до 1
    else:
        accuracy = 1.0
    
    # Сохраняем в историю
    metrics_history['simple_count'].append(accuracy)
    
    return accuracy

def calculate_iou_basic_metric(model, dataloader, device, stage, epoch):
    """
    МЕТРИКА 2: IoU Basic - простая метрика на основе IoU
    """
    model.eval()    
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]            
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                above_threshold = pred['scores'] > DETECT_THRESHOLD
                pred_boxes = pred['boxes'][above_threshold]
                true_boxes = target['boxes'].to(device)

                pred_count = len(pred_boxes)
                true_count = len(true_boxes)

                # Случай 1: Нет предсказаний
                if pred_count == 0 and true_count > 0:
                    total_fn += true_count
                # Случай 2: Нет истинных bbox'ов  
                elif pred_count > 0 and true_count == 0:
                    total_fp += pred_count
                # Случай 3: Есть и pred, и true
                elif pred_count > 0 and true_count > 0:
                    iou_matrix = box_iou(pred_boxes, true_boxes)
                    
                    # Простой matching
                    for true_idx in range(true_count):
                        best_iou = torch.max(iou_matrix[:, true_idx]).item()
                        if best_iou >= IOU_TRESHHOLD:
                            total_tp += 1
                        else:
                            total_fn += 1
                    
                    total_fp += max(0, pred_count - true_count)

    # Accuracy = TP / (TP + FP + FN)
    if (total_tp + total_fp + total_fn) > 0:
        accuracy = total_tp / (total_tp + total_fp + total_fn)
    else:
        accuracy = 0.0
    
    metrics_history['iou_basic'].append(accuracy)
    
    return accuracy

def calculate_iou_advanced_metric(model, dataloader, device, stage, epoch):
    """
    МЕТРИКА 3: IoU Advanced - продвинутая метрика с bipartite matching
    """
    model.eval()    
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]            
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                above_threshold = pred['scores'] > DETECT_THRESHOLD
                pred_boxes = pred['boxes'][above_threshold]
                true_boxes = target['boxes'].to(device)

                pred_count = len(pred_boxes)
                true_count = len(true_boxes)

                if pred_count == 0 and true_count > 0:
                    total_fn += true_count
                elif pred_count > 0 and true_count == 0:
                    total_fp += pred_count
                elif pred_count > 0 and true_count > 0:
                    iou_matrix = box_iou(pred_boxes, true_boxes)
                    
                    # Bipartite matching
                    used_preds = set()
                    used_trues = set()
                    
                    # Собираем все возможные пары с хорошим IoU
                    good_pairs = []
                    for pred_idx in range(pred_count):
                        for true_idx in range(true_count):
                            iou_val = iou_matrix[pred_idx, true_idx].item()
                            if iou_val >= IOU_TRESHHOLD:
                                good_pairs.append((iou_val, pred_idx, true_idx))
                    
                    # Сортируем по IoU (лучшие first)
                    good_pairs.sort(reverse=True, key=lambda x: x[0])
                    
                    # Жадный matching
                    for iou_val, pred_idx, true_idx in good_pairs:
                        if pred_idx not in used_preds and true_idx not in used_trues:
                            total_tp += 1
                            used_preds.add(pred_idx)
                            used_trues.add(true_idx)
                    
                    total_fp += pred_count - len(used_preds)
                    total_fn += true_count - len(used_trues)

    if (total_tp + total_fp + total_fn) > 0:
        accuracy = total_tp / (total_tp + total_fp + total_fn)
    else:
        accuracy = 0.0
    
    metrics_history['iou_advanced'].append(accuracy)
    
    return accuracy

def calculate_all_metrics(model, dataloader, device, stage, epoch):
    """
    Вычисляет все 3 метрики и строит графики
    """
    print(f"📊 Вычисление метрик - Stage {stage}, Epoch {epoch+1}")
    
    # Вычисляем все метрики
    accuracy_simple = calculate_simple_count_metric(model, dataloader, device, stage, epoch)
    accuracy_iou_basic = calculate_iou_basic_metric(model, dataloader, device, stage, epoch)
    accuracy_iou_advanced = calculate_iou_advanced_metric(model, dataloader, device, stage, epoch)
    
    print(f"   Simple Count: {accuracy_simple:.4f}")
    print(f"   IoU Basic: {accuracy_iou_basic:.4f}")
    print(f"   IoU Advanced: {accuracy_iou_advanced:.4f}")
    
    # Строим графики для каждой метрики
    epochs = range(1, epoch + 2)  # Эпохи от 1 до текущей
    
    for metric_name in metrics_history.keys():
        if len(metrics_history[metric_name]) > 0:
            values = metrics_history[metric_name]
            plot_metric_per_epoch(metric_name, stage, epochs, values)
    
    # Строим график сравнения
    plot_all_metrics_comparison(stage, metrics_history)
    
    # Возвращаем основную метрику (IoU Advanced)
    return accuracy_iou_advanced

def get_metrics_history():
    """Возвращает историю всех метрик"""
    return metrics_history.copy()

def reset_metrics_history():
    """Сбрасывает историю метрик"""
    global metrics_history
    metrics_history = {
        'simple_count': [],
        'iou_basic': [], 
        'iou_advanced': []
    }