"""
Модуль для загрузки и распаковки датасета
"""
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
from src.config import ARCHIVE_PATH, DATA_ROOT, EXTRACTED_DATA_PATH

def download_zenodo_zip():
    """
    Скачивает датасет с Zenodo
    
    Returns:
        Path: путь к скачанному архиву
    """
    url = 'https://zenodo.org/records/14185684/files/VME_CDSI_datasets.zip?download=1'
    
    print("Найден zip:", url)
    
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        
        with open(ARCHIVE_PATH, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True, desc='Downloading'
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return ARCHIVE_PATH

def extract_archive():
    """
    Распаковывает архив с данными
    
    Returns:
        bool: True если распаковка успешна
    """
    if not ARCHIVE_PATH.exists():
        print(f"Архив не найден: {ARCHIVE_PATH}")
        return False
    
    print("Распаковываю архив...")
    with zipfile.ZipFile(ARCHIVE_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_ROOT)
    
    print(f"Данные распакованы в: {EXTRACTED_DATA_PATH}")
    return True

def download_and_extract_data():
    """
    Основная функция для загрузки и подготовки данных
    """
    # Проверяем, существует ли уже распакованные данные
    if EXTRACTED_DATA_PATH.exists():
        print("Данные уже распакованы:", EXTRACTED_DATA_PATH)
        return True
    
    # Скачиваем если нужно
    if not ARCHIVE_PATH.exists():
        print("Скачиваю данные...")
        try:
            download_zenodo_zip()
            print("Скачано:", ARCHIVE_PATH)
        except Exception as e:
            print("Ошибка при загрузке с Zenodo:", e)
            print(f"Пожалуйста, скачайте архив вручную и поместите в: {ARCHIVE_PATH}")
            return False
    
    # Распаковываем
    return extract_archive()