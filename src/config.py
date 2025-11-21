"""
Конфигурационные параметры для обучения Faster R-CNN
"""
from pathlib import Path
import torch

# Пути к данным
DATA_ROOT = Path("./VME_data").absolute()
DATA_ROOT.mkdir(parents=True, exist_ok=True)
ARCHIVE_PATH = DATA_ROOT / "VME_CDSI_datasets.zip"
EXTRACTED_DATA_PATH = DATA_ROOT / "VME_CDSI_datasets"

# === НАСТРОЙКИ ДЛЯ ОГРАНИЧЕНИЯ ДАННЫХ ===
DEBUG_MODE = True  # Включить режим отладки
MAX_SAMPLES = 10   # Ограничить количество изображений до 10
# =======================================

# Настройки Anchors
ANCHOR_SIZES = (((8,), (16,), (32,), (64,)))
ANCHOR_RATIOS = ((0.8, 1.0, 1.2),) * 4    

# Гиперпараметры обучения
BATCH_SIZE = 4
LEARNING_RATE_STAGE_A = 0.005
LEARNING_RATE_STAGE_B = 0.0005
NUM_EPOCHS_STAGE_A = 5
NUM_EPOCHS_STAGE_B = 3
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
STEP_SIZE = 3
GAMMA = 0.1

# Пороги согласий
DETECT_THRESHOLD = 0.5
IOU_TRESHHOLD = 0.5

# Настройки устройства
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Пути к данным после распаковки
def get_data_paths():
    """Возвращает пути к изображениям и аннотациям"""
    img_dir = EXTRACTED_DATA_PATH / "satellite_images"
    ann_dir = EXTRACTED_DATA_PATH / "annotations_HBB"
    train_ann = ann_dir / "train.json"
    val_ann = ann_dir / "val.json"
    return img_dir, train_ann, val_ann

# Настройки модели
PRETRAINED_BACKBONE = True