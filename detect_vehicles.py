"""
Скрипт для детекции транспорта на изображениях и сохранения с рамками
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from pycocotools.coco import COCO

from src.models.faster_rcnn import get_model
from src.config import DEVICE, THRESHOLD, get_data_paths

def get_num_classes(annotation_path):
    """
    Получает количество классов из COCO аннотаций
    """
    coco = COCO(annotation_path)
    cat_ids = coco.getCatIds()
    return 1 + len(cat_ids)  # +1 для background класса

def load_trained_model():
    """Загружает обученную модель"""
    print("Загрузка модели...")
    
    img_dir, train_ann, val_ann = get_data_paths()
    num_classes = get_num_classes(train_ann)
    
    model = get_model(num_classes)
    model.load_state_dict(torch.load('VME_data/models/final_faster_rcnn.pth', map_location='cpu'))
    model.to(DEVICE)
    model.eval()
    
    print(f"Модель загружена! Классы: {num_classes}")
    return model, num_classes

def detect_vehicles(model, image_path, confidence_threshold=THRESHOLD):
    """Детектирует транспорт на одном изображении"""    
    
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)    
    
    with torch.no_grad():
        predictions = model(image_tensor)  
    
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    
    # Фильтруем по confidence
    keep = scores >= confidence_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    print(f"Найдено объектов: {len(boxes)}")
    
    return image, boxes, scores, labels

def draw_boxes(image, boxes, scores, labels, output_path):
    """Рисует рамки на изображении и сохраняет"""
    
    # Конвертируем PIL в OpenCV для рисования
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):        
        x1, y1, x2, y2 = map(int, box)        
        
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)        
        
        label_text = f"Obj {label}: {score:.2f}"
        cv2.putText(image_cv, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    
    
    cv2.imwrite(str(output_path), image_cv)
    print(f"Сохранено: {output_path}")

def find_image_path_by_id(img_dir, image_id):
    """Находит путь к изображению по его COCO ID"""    
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_file in img_dir.glob(ext):
            if str(image_id) in img_file.stem:
                return img_file
    
    # Если не нашли по ID, попробуем получить по индексу через COCO
    try:
        from src.data.dataset import CocoDetectionForFasterRCNN
        _, train_ann, val_ann = get_data_paths()
        coco = COCO(val_ann)
        img_info = coco.loadImgs(image_id)[0]
        image_path = img_dir / img_info['file_name']
        if image_path.exists():
            return image_path
    except:
        pass
    
    return None

def main(images_to_process):
    """Основная функция детекции"""
    print("Детекция объектов на изображениях")
    print("=" * 50)    
    
    output_dir = Path("detection_results")
    output_dir.mkdir(exist_ok=True)    
    
    model, num_classes = load_trained_model()    
    
    img_dir, train_ann, val_ann = get_data_paths()    
    
    from src.data.dataset import CocoDetectionForFasterRCNN
    val_dataset = CocoDetectionForFasterRCNN(root=str(img_dir), annFile=str(val_ann))
    
    print(f"Всего изображений в валидации: {len(val_dataset)}")    
    
    num_images = min(images_to_process, len(val_dataset))
    
    for i in range(num_images):
        print(f"\n Обработка изображения {i+1}/{num_images}...")
        
        try:
            # Получаем изображение и его ID из датасета
            image_tensor, target = val_dataset[i]
            image_id = target['image_id'].item()            
            
            image_path = find_image_path_by_id(img_dir, image_id)
            
            if image_path and image_path.exists():
                print(f"   Изображение: {image_path.name}")                
                
                image, boxes, scores, labels = detect_vehicles(model, image_path)                
                
                output_path = output_dir / f"detected_{i+1:03d}.jpg"
                draw_boxes(image, boxes, scores, labels, output_path)
                
                print(f"   Обнаружено объектов: {len(boxes)}")
            else:
                print(f"   Изображение не найдено для ID: {image_id}")
                
        except Exception as e:
            print(f"   Ошибка при обработке: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nГотово! Результаты сохранены в: {output_dir}")

if __name__ == "__main__":
    main(30)