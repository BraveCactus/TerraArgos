"""
Считает среднюю accuracy по эпохам из CSV файлов и строит графики
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_DIR = Path("results/classic/acc")
PLOTS_DIR = Path("results/classic/graphics/acc")

def analyze_metrics():
    """Основная функция анализа"""
    print("АНАЛИЗ МЕТРИК ИЗ CSV ФАЙЛОВ")
    print("=" * 40)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(CSV_DIR.glob("*.csv"))
    print(f"📁 Найдено CSV файлов: {len(csv_files)}")
    
    if not csv_files:
        print("Нет CSV файлов для анализа")
        return

    stage_a_data = []
    stage_b_data = []
    
    for csv_file in csv_files:
        try:           
            df = pd.read_csv(csv_file)            
            
            mean_acc = df['accuracy'].mean()
            
            filename = csv_file.stem
            if 'stage_A' in filename:
                stage = 'A'
            elif 'stage_B' in filename:
                stage = 'B'
            else:
                stage = 'Unknown'            
            
            epoch_match = re.search(r'epoch_(\d+)', filename)
            if epoch_match:
                epoch = int(epoch_match.group(1))
            else:
                epoch = None
            
            if stage == 'A' and epoch:
                stage_a_data.append((epoch, mean_acc))
                print(f"Stage A, Epoch {epoch}: accuracy = {mean_acc:.4f}")
            elif stage == 'B' and epoch:
                stage_b_data.append((epoch, mean_acc))
                print(f"Stage B, Epoch {epoch}: accuracy = {mean_acc:.4f}")
                
        except Exception as e:
            print(f"Ошибка в файле {csv_file.name}: {e}")

    stage_a_data.sort(key=lambda x: x[0])
    stage_b_data.sort(key=lambda x: x[0])

    plot_accuracy_by_epoch(stage_a_data, stage_b_data)

def plot_accuracy_by_epoch(stage_a_data, stage_b_data):
    """Строит график accuracy по эпохам"""
    plt.figure(figsize=(12, 6))

    if stage_a_data:
        epochs_a = [x[0] for x in stage_a_data]
        accuracies_a = [x[1] for x in stage_a_data]
        plt.plot(epochs_a, accuracies_a, 'bo-', linewidth=2, markersize=6, label='Stage A')
        
        for i, (epoch, acc) in enumerate(stage_a_data):
            plt.annotate(f'{acc:.3f}', (epoch, acc), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=8)

    if stage_b_data:
        epochs_b = [x[0] for x in stage_b_data]
        accuracies_b = [x[1] for x in stage_b_data]
        plt.plot(epochs_b, accuracies_b, 'ro-', linewidth=2, markersize=6, label='Stage B')

        for i, (epoch, acc) in enumerate(stage_b_data):
            plt.annotate(f'{acc:.3f}', (epoch, acc),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=8)
    
    plt.title('Accuracy по эпохам обучения', fontsize=14, fontweight='bold')
    plt.xlabel('Эпоха')
    plt.ylabel('Средняя Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    all_epochs = []
    if stage_a_data:
        all_epochs.extend([x[0] for x in stage_a_data])
    if stage_b_data:
        all_epochs.extend([x[0] for x in stage_b_data])
    
    if all_epochs:
        plt.xticks(all_epochs)

    plot_path = PLOTS_DIR / "accuracy_by_epoch.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nГрафик сохранен: {plot_path}")

    save_epoch_data(stage_a_data, stage_b_data)

def save_epoch_data(stage_a_data, stage_b_data):
    """Сохраняет средние accuracy по эпохам в CSV"""
    csv_data = []

    for epoch, accuracy in stage_a_data:
        csv_data.append({
            'stage': 'A',
            'epoch': epoch,
            'mean_accuracy': accuracy
        })

    for epoch, accuracy in stage_b_data:
        csv_data.append({
            'stage': 'B',
            'epoch': epoch,
            'mean_accuracy': accuracy
        })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = PLOTS_DIR / "epoch_accuracy_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"Данные сохранены: {csv_path}")

        print("\nСТАТИСТИКА:")
        if stage_a_data:
            max_a = max([x[1] for x in stage_a_data])
            print(f"   Stage A: макс. accuracy = {max_a:.4f}")
        if stage_b_data:
            max_b = max([x[1] for x in stage_b_data])
            print(f"   Stage B: макс. accuracy = {max_b:.4f}")

if __name__ == "__main__":
    analyze_metrics()