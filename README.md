## Структура сохраняемых данных

VME_data/
├── models/                           # Сохраненные модели
│   ├── stage_a_epoch_1.pth
│   ├── stage_a_epoch_2.pth
│   ├── stage_b_epoch_1.pth
│   └── final_faster_rcnn.pth
└── metrics_plots/                    # Все графики метрик
    ├── simple_count/                 # Метрика 1: сравнение количества
    │   ├── simple_count_stage_A.png
    │   └── simple_count_stage_B.png
    ├── iou_basic/                    # Метрика 2: базовая IoU
    │   ├── iou_basic_stage_A.png
    │   └── iou_basic_stage_B.png
    ├── iou_advanced/                 # Метрика 3: продвинутая IoU (основная)
    │   ├── iou_advanced_stage_A.png
    │   └── iou_advanced_stage_B.png
    ├── comparison/                   # Сравнение всех метрик
    │   ├── comparison_stage_A.png
    │   └── comparison_stage_B.png
    └── training_progress/            # Прогресс обучения
        ├── progress_stage_A.png
        └── progress_stage_B.png