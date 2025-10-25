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

    total_tp = 0  # True Positive
    total_fp = 0  # False Positive  
    total_fn = 0  # False Negative

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

                image_tp = 0
                image_tn = 0
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

                total_tp += image_tp
                total_fp += image_fp
                total_fn += image_fn
        
        if (total_tp + total_fp + total_fn) > 0:
            accuracy = total_tp / (total_tp + total_fp + total_fn)
        else:
            accuracy = 0.0
        
        return accuracy
