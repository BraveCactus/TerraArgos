import torch
from torchvision.ops import box_iou
from src.config import DETECT_THRESHOLD, IOU_TRESHHOLD


def calculate_metrics(model, dataloader, device, detect_threshhold=DETECT_THRESHOLD, iou_treshold=IOU_TRESHHOLD):
    """
    Считает метрики для изображений из датасета

    Args: 
        model: предобученная модель
        dataloader: объект Dataloader с датасетом
        device: cpu/cuda
        detect_threshhold: порог согласия с классом объекта
        iou_treshold: порог согласия с точностью определения прямоугольника (box)
    """
    model.eval()

    predictions = model(images)

    correct_predictions = 0
    total_predictions = 0
    missed_objects = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]

            predictions = model(images)

            for i, (pred, target) in enumerate(zip(predictions, targets)):
                above_threshold = pred['scores'] > detect_threshhold
                
                pred_boxes = pred['boxes'][above_threshold]
                true_boxes = target['boxes'].to(device)                

                pred_vehicles = len(pred_boxes)
                true_vehicles = len(true_boxes)

                # FN
                if (pred_vehicles == 0 and true_vehicles > 0):
                    missed_objects += true_vehicles
                    continue

                # FP
                if (pred_vehicles > 0 and true_vehicles == 0):
                    total_predictions += pred_vehicles
                    continue

                # TP+TN
                if (pred_vehicles > 0 and true_vehicles > 0):
                    iou_matrix = box_iou(pred_boxes, true_boxes)

                    for true_idx in range(len(true_boxes)):
                        best_iou = torch.max(iou_matrix[:, true_idx]).item()
                        if best_iou > iou_treshold:
                            correct_predictions += 1
    
                    
                    total_predictions += len(pred_boxes)                    
                    missed_objects += max(0, len(true_boxes) - correct_predictions)    
        
        accuracy = correct_predictions / (total_predictions + missed_objects)
        
        return accuracy
