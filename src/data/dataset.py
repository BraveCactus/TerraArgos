"""
Кастомный датасет для COCO в формате Faster R-CNN
"""
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
import torch
from pycocotools.coco import COCO

class CocoDetectionForFasterRCNN(CocoDetection):
    """
    Наследует от CocoDetection и преобразует данные в формат Faster R-CNN
    """
    
    def __getitem__(self, idx):
        """
        Получает элемент датасета и преобразует в формат Faster R-CNN
        
        Args:
            idx (int): индекс элемента
            
        Returns:
            tuple: (image_tensor, target_dict)
        """
        # Получаем оригинальные данные COCO
        img, target = super().__getitem__(idx)
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        # Получаем оригинальный COCO image ID
        img_id = self.ids[idx]

        # Преобразуем каждый объект в аннотации
        for obj in target:
            x, y, w, h = obj["bbox"]
            # Конвертируем COCO [x, y, width, height] в Pascal VOC [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(obj["category_id"])
            areas.append(obj.get("area", w * h))  # Используем area если есть, иначе вычисляем
            iscrowd.append(obj.get("iscrowd", 0))

        # Конвертируем в тензоры
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # Создаем словарь целей в формате Faster R-CNN
        target_out = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        # Конвертируем изображение в тензор
        img = F.to_tensor(img)
        
        return img, target_out

def get_num_classes(annotation_path):
    """
    Получает количество классов из COCO аннотаций
    
    Args:
        annotation_path (str): путь к JSON файлу с аннотациями
        
    Returns:
        int: количество классов (включая background)
    """
    coco = COCO(annotation_path)
    cat_ids = coco.getCatIds()
    return 1 + len(cat_ids)  # +1 для background класса

def get_names_classes(annotation_path):
    """
    Получает присутсвующие в датасете классы
    Args:
        annotation_path (str): путь к JSON файлу с аннотациями
    
    Returns:
        dict[str, int]: словарь классов {имя_класса: id_класса}
    """
    coco = COCO(annotation_path)
    cat_ids = coco.getCatIds()
    categories = coco.loadCats(cat_ids)
    cat_names = [cat['name'] for cat in categories]

    cat_dict = {name: id for name, id in zip(cat_names, cat_ids)}    

    return cat_dict