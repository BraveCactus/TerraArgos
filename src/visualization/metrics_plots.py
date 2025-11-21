"""
Построение графиков для каждой метрики отдельно
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src.config import DATA_ROOT

def plot_metric_per_epoch(metric_name, stage, epochs, values):
    """
    Строит график одной метрики по эпохам для конкретной стадии
    """
    plt.figure(figsize=(8, 5))
    
    plt.plot(epochs, values, marker='o', linewidth=2, markersize=6)
    plt.title(f'{metric_name} - Stage {stage}')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    
    # Сохраняем в папку для этой метрики
    plots_dir = Path(f"{DATA_ROOT}/metrics_plots/{metric_name}")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / f"{metric_name}_stage_{stage}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 {metric_name} stage {stage}: {plot_path}")

def plot_all_metrics_comparison(stage, metrics_data):
    """
    Строит график сравнения всех метрик для одной стадии
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(next(iter(metrics_data.values()))) + 1)
    
    for metric_name, values in metrics_data.items():
        if values:  # Если есть данные
            plt.plot(epochs, values, marker='s', linewidth=2, label=metric_name)
    
    plt.title(f'Сравнение метрик - Stage {stage}', fontweight='bold')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    
    # Сохраняем в папку сравнения
    plots_dir = Path(f"{DATA_ROOT}/metrics_plots/comparison")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / f"comparison_stage_{stage}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Сравнение stage {stage}: {plot_path}")

def plot_training_progress(loss_history, accuracy_history, stage):
    """
    Строит график прогресса обучения (loss + accuracy)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # График Loss
    if loss_history:
        epochs = range(1, len(loss_history) + 1)
        ax1.plot(epochs, loss_history, marker='o', color='red', linewidth=2)
        ax1.set_title(f'Loss - Stage {stage}')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(epochs)
    
    # График Accuracy
    if accuracy_history:
        epochs = range(1, len(accuracy_history) + 1)
        ax2.plot(epochs, accuracy_history, marker='o', color='blue', linewidth=2)
        ax2.set_title(f'Accuracy - Stage {stage}')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(epochs)
    
    # Сохраняем
    plots_dir = Path(f"{DATA_ROOT}/metrics_plots/training_progress")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / f"progress_stage_{stage}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Прогресс stage {stage}: {plot_path}")