"""
Модуль для создания модели Faster R-CNN
"""
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from src.config import PRETRAINED_BACKBONE
from .anchors import create_anchor_generator

def get_model(num_classes, pretrained_backbone=PRETRAINED_BACKBONE):
    """
    Создает модель Faster R-CNN с ResNet-50 backbone
    
    Args:
        num_classes (int): количество классов (включая background)
        pretrained_backbone (bool): использовать предобученный backbone
        
    Returns:
        torchvision.models.detection.FasterRCNN: модель Faster R-CNN
    """
    # Загружаем предобученную модель
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    
    # Получаем количество входных features для классификатора
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Заменяем классификатор на новый с нашим количеством классов
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, 
        num_classes
    )
    
    return model

def freeze_backbone(model):
    """
    Замораживает веса backbone сети
    
    Args:
        model: модель Faster R-CNN
    """
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad = False
    print("Backbone заморожен")

def unfreeze_backbone(model):
    """
    Размораживает веса backbone сети
    
    Args:
        model: модель Faster R-CNN
    """
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad = True
    print("Backbone разморожен")


def get_model_with_anchors(num_classes, anchor_sizes=((32,), (64,), (128,), (256,)), 
                          anchor_ratios=((0.5, 1.0, 2.0),) * 4):
    """
    Простая версия с кастомными anchor'ами
    
    Args:
        num_classes: количество классов
        anchor_sizes: размеры anchor'ов
        anchor_ratios: формы anchor'ов
    """   
    
    # Берем стандартную модель
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    
    # Меняем только anchor generator
    model.rpn.anchor_generator = create_anchor_generator(anchor_sizes, anchor_ratios)
    
    # Обновляем классификатор (как было)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    return model