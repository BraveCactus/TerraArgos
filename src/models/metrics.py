from pathlib import Path
from matplotlib import pyplot as plt
import torch
import pandas as pd
from torchvision.ops import box_iou
from src.config import DETECT_THRESHOLD, IOU_TRESHHOLD, RESULTS_ROOT


def calculate_metrics(model_name, model, dataloader, device, stage, epoch, 
                     detect_threshold=DETECT_THRESHOLD, iou_threshold=IOU_TRESHHOLD):
    """
    Считает метрики (accuracy, precision, recall, f1) детекции для изображений из датасета для заданной модели в определенной эпохе обучения.

    Args: 
        model_name: название модели
        model: предобученная модель
        dataloader: объект Dataloader с датасетом
        device: cpu/cuda
        stage: стадия обучения ('A' или 'B')
        epoch: номер эпохи
        detect_threshold: порог уверенности для предсказаний
        iou_threshold: порог IoU для сопоставления bounding boxes

    Returns:
        dict: словарь с метриками {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}
    """
    model.eval()    

    total_tp = 0  
    total_fp = 0 
    total_fn = 0  
    
    batch_metrics = []
    image_metrics = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]            
            predictions = model(images)
            
            batch_tp = batch_fp = batch_fn = 0            
            
            for img_idx, (pred, target) in enumerate(zip(predictions, targets)):

                above_threshold = pred['scores'] > detect_threshold
                pred_boxes = pred['boxes'][above_threshold]
                true_boxes = target['boxes'].to(device)                
                
                pred_count = len(pred_boxes)
                true_count = len(true_boxes)
                
                image_tp = image_fp = image_fn = 0                
                
                if pred_count == 0 and true_count == 0:                    
                    continue                
                
                elif pred_count == 0 and true_count > 0:
                    image_fn = true_count                
                
                elif pred_count > 0 and true_count == 0:
                    image_fp = pred_count                
                
                else:
                    iou_matrix = box_iou(pred_boxes, true_boxes)
                    
                    matched_preds = set()
                    matched_trues = set()                    
                    
                    for i in range(pred_count):
                        for j in range(true_count):
                            if i in matched_preds or j in matched_trues:
                                continue
                            if iou_matrix[i, j] >= iou_threshold:
                                image_tp += 1
                                matched_preds.add(i)
                                matched_trues.add(j)
                    
                    image_fp = pred_count - len(matched_preds)
                    image_fn = true_count - len(matched_trues)                
                
                image_total = image_tp + image_fp + image_fn
                image_accuracy = image_tp / image_total if image_total > 0 else 0.0
                image_precision = image_tp / (image_tp + image_fp) if (image_tp + image_fp) > 0 else 0.0
                image_recall = image_tp / (image_tp + image_fn) if (image_tp + image_fn) > 0 else 0.0
                image_f1 = 2 * image_precision * image_recall / (image_precision + image_recall) \
                          if (image_precision + image_recall) > 0 else 0.0                
                
                image_metrics.append({
                    'batch_num': batch_idx,
                    'image_in_batch': img_idx,
                    'stage': stage,
                    'epoch': epoch + 1,
                    'pred_count': pred_count,
                    'true_count': true_count,
                    'tp': image_tp,
                    'fp': image_fp,
                    'fn': image_fn,
                    'accuracy': image_accuracy,
                    'precision': image_precision,
                    'recall': image_recall,
                    'f1': image_f1
                })                
                
                batch_tp += image_tp
                batch_fp += image_fp
                batch_fn += image_fn            
            
            batch_total = batch_tp + batch_fp + batch_fn
            batch_accuracy = batch_tp / batch_total if batch_total > 0 else 0.0
            batch_precision = batch_tp / (batch_tp + batch_fp) if (batch_tp + batch_fp) > 0 else 0.0
            batch_recall = batch_tp / (batch_tp + batch_fn) if (batch_tp + batch_fn) > 0 else 0.0
            batch_f1 = 2 * batch_precision * batch_recall / (batch_precision + batch_recall) \
                      if (batch_precision + batch_recall) > 0 else 0.0
            
            batch_metrics.append({
                'batch_num': batch_idx,
                'images_in_batch': len(images),
                'tp': batch_tp,
                'fp': batch_fp,
                'fn': batch_fn,
                'accuracy': batch_accuracy,
                'precision': batch_precision,
                'recall': batch_recall,
                'f1': batch_f1
            })
            
            
            total_tp += batch_tp
            total_fp += batch_fp
            total_fn += batch_fn
    
    
    total_detections = total_tp + total_fp + total_fn
    accuracy = total_tp / total_detections if total_detections > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0    

    current_summary = {
        'model': model_name,
        'stage': stage,
        'epoch': epoch + 1,
        'total_images': len(image_metrics),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_pred_per_image': sum(m['pred_count'] for m in image_metrics) / len(image_metrics) if image_metrics else 0,
        'avg_true_per_image': sum(m['true_count'] for m in image_metrics) / len(image_metrics) if image_metrics else 0
    }
    
    try:
        results_dir = RESULTS_ROOT / f"{model_name}_results" / "metrics"        
        results_dir.mkdir(parents=True, exist_ok=True)

        # 1. Сохраняем batch metrics в отдельный файл
        batch_metrics_dir = results_dir / "batch_metrics"
        batch_metrics_dir.mkdir(parents=True, exist_ok=True)
        
        batch_df = pd.DataFrame(batch_metrics)
        batch_file = batch_metrics_dir / f"stage_{stage}_epoch_{epoch+1}_batches.csv"
        batch_df.to_csv(batch_file, index=False)        
        
        # 2. Сохраняем image metrics в отдельный файл (опционально)
        image_metrics_dir = results_dir / "image_metrics"
        image_metrics_dir.mkdir(parents=True, exist_ok=True)
        
        image_df = pd.DataFrame(image_metrics)
        image_file = image_metrics_dir / f"stage_{stage}_epoch_{epoch+1}_images.csv"
        image_df.to_csv(image_file, index=False)        
        
        summary_file = results_dir / "all_summaries.csv"

        if summary_file.exists():            
            existing_df = pd.read_csv(summary_file)            
            
            mask = (existing_df['model'] == model_name) & \
                   (existing_df['stage'] == stage) & \
                   (existing_df['epoch'] == epoch + 1)
            
            if mask.any():                
                existing_df.loc[mask, list(current_summary.keys())] = list(current_summary.values())
            else:                
                new_row = pd.DataFrame([current_summary])
                existing_df = pd.concat([existing_df, new_row], ignore_index=True)
        else:            
            existing_df = pd.DataFrame([current_summary])        
       
        existing_df.to_csv(summary_file, index=False)
        
        print(f"Метрики сохранены в {results_dir}/")
        print(f"   Изображений обработано: {len(image_metrics)}")       
        
    except Exception as e:
        print(f"Ошибка при сохранении метрик: {e}")       
    
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'total_images': len(image_metrics)
    }

def plot_training_metrics(model_name):
    """
    Aункция для построения 4 графиков: precision, recall, f1, loss
    """    
    metrics_file = RESULTS_ROOT / f"{model_name}_results" / "metrics" / "all_summaries.csv"
    losses_file = RESULTS_ROOT / f"{model_name}_results" / "training_losses" / "all_losses.csv"    
    
    metrics = pd.read_csv(metrics_file)
    losses = pd.read_csv(losses_file)    
    
    plots_dir = RESULTS_ROOT / f"{model_name}_results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. График Recall
    plt.figure(figsize=(10, 6))
    for stage in metrics['stage'].unique():
        stage_data = metrics[metrics['stage'] == stage]
        plt.plot(stage_data['epoch'], stage_data['recall'], marker='o', label=f'Stage {stage}')
    plt.title(f'Recall - {model_name}')
    plt.xlabel('Эпоха')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "recall.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # 2. График Precision
    plt.figure(figsize=(10, 6))
    for stage in metrics['stage'].unique():
        stage_data = metrics[metrics['stage'] == stage]
        plt.plot(stage_data['epoch'], stage_data['precision'], marker='s', label=f'Stage {stage}')
    plt.title(f'Precision - {model_name}')
    plt.xlabel('Эпоха')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "precision.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # 3. График F1-score
    plt.figure(figsize=(10, 6))
    for stage in metrics['stage'].unique():
        stage_data = metrics[metrics['stage'] == stage]
        plt.plot(stage_data['epoch'], stage_data['f1'], marker='^', label=f'Stage {stage}')
    plt.title(f'F1-score - {model_name}')
    plt.xlabel('Эпоха')
    plt.ylabel('F1-score')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "f1_score.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # 4. График Loss
    plt.figure(figsize=(10, 6))
    for stage in losses['stage'].unique():
        stage_data = losses[losses['stage'] == stage]
        plt.plot(stage_data['epoch'], stage_data['avg_loss'], marker='D', label=f'Stage {stage}')
    plt.title(f'Loss - {model_name}')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "loss.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Графики сохранены в: {plots_dir}")
