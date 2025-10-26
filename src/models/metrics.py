from pathlib import Path
import torch
import pandas as pd
from torchvision.ops import box_iou
from src.config import BATCH_SIZE, DETECT_THRESHOLD, IOU_TRESHHOLD, DATA_ROOT


def calculate_metrics(model, dataloader, device, stage, epoch, detect_threshhold=DETECT_THRESHOLD, iou_treshold=IOU_TRESHHOLD):
    """
    Считает метрики для изображений из датасета

    Args: 
        model: предобученная модель
        dataloader: объект Dataloader с датасетом
        device: cpu/cuda
        detect_threshhold: порог согласия с классом объекта
        iou_treshold: порог согласия с точностью определения прямоугольника (box)

    Returns:
        accuracy: метрика для эпохи
    """
    model.eval()    

    total_tp = 0  # True Positive
    total_fp = 0  # False Positive  
    total_fn = 0  # False Negative

    batch_accuracies = list()

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]            

            predictions = model(images)

            batch_tp = 0 
            batch_fp = 0  
            batch_fn = 0

            for i, (pred, target) in enumerate(zip(predictions, targets)):
                above_threshold = pred['scores'] > detect_threshhold
                
                pred_boxes = pred['boxes'][above_threshold]
                true_boxes = target['boxes'].to(device)                

                pred_vehicles = len(pred_boxes)
                true_vehicles = len(true_boxes)

                image_tp = 0                
                image_fp = 0
                image_fn = 0  

                # FN
                if (pred_vehicles == 0 and true_vehicles > 0):
                    image_fn += true_vehicles                    

                # FP
                elif (pred_vehicles > 0 and true_vehicles == 0):
                    image_fp += pred_vehicles   

                # TP
                elif (pred_vehicles > 0 and true_vehicles > 0):
                    iou_matrix = box_iou(pred_boxes, true_boxes)

                    for true_idx in range(len(true_boxes)):
                        best_iou = torch.max(iou_matrix[:, true_idx]).item()
                        if best_iou >= iou_treshold:
                            image_tp += 1
    
                    
                    image_fp = pred_vehicles - image_tp  
                    image_fn = true_vehicles - image_tp

                # Накопление метрик для батча
                batch_tp += image_tp
                batch_fp += image_fp
                batch_fn += image_fn

            # Расчет метрик для текущего батча
            if (batch_tp + batch_fp + batch_fn) > 0:
                batch_accuracy = batch_tp / (batch_tp + batch_fp + batch_fn)
            else:
                batch_accuracy = 0.0  

            batch_accuracies.append(batch_accuracy)              

            total_tp += batch_tp
            total_fp += batch_fp
            total_fn += batch_fn

        # Сохраняем метрики всех батчей
        try:
            accuracy_df = pd.DataFrame({
                "batch_num": range(len(batch_accuracies)),
                "accuracy": batch_accuracies
            })

            accur_dir_path = Path(f"{DATA_ROOT}/accuracy")
            accur_dir_path.mkdir(parents=True, exist_ok=True)
            accuracy_file_path = accur_dir_path / f"accuracy_stage_{stage}_epoch_{epoch+1}.csv"
            accuracy_df.to_csv(accuracy_file_path, index=False)

            print(f"Метрики батчей сохранены в {accuracy_file_path}")
        except Exception as e:
            print(f"Ошибка при сохранении метрик в {accuracy_file_path}: {e}")

        # Общая метрика по всем данным
        if (total_tp + total_fp + total_fn) > 0:
            accuracy = total_tp / (total_tp + total_fp + total_fn)
        else:
            accuracy = 0.0
        
    return accuracy