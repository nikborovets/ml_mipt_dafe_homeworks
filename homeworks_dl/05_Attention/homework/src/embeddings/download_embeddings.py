#!/usr/bin/env python3
"""
Скрипт для загрузки предобученных русских эмбеддингов FastText.

Источник: https://fasttext.cc/docs/en/crawl-vectors.html
Russian: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz
"""

import os
import requests
import gzip
import shutil
from pathlib import Path
from tqdm.auto import tqdm


def download_file(url, output_path):
    """Загружает файл с прогресс-баром."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(url)}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)


def extract_gz(gz_path, output_path):
    """Извлекает .gz файл."""
    print(f"Extracting {gz_path} -> {output_path}")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def main():
    """Основная функция загрузки русских эмбеддингов."""
    # URLs для русских эмбеддингов FastText
    urls = {
        'cc.ru.300.bin.gz': 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz',
        'cc.ru.300.vec.gz': 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz'
    }
    
    embeddings_dir = Path(__file__).parent
    embeddings_dir.mkdir(exist_ok=True)
    
    for filename, url in urls.items():
        gz_path = embeddings_dir / filename
        bin_path = embeddings_dir / filename.replace('.gz', '')
        
        # Проверяем, не скачан ли уже файл
        if bin_path.exists():
            print(f"✓ {bin_path.name} already exists, skipping download")
            continue
            
        if not gz_path.exists():
            print(f"Downloading {filename}...")
            download_file(url, gz_path)
        
        # Извлекаем файл
        extract_gz(gz_path, bin_path)
        
        # Удаляем архив для экономии места
        gz_path.unlink()
        print(f"✓ {bin_path.name} extracted successfully")
    
    print("✓ All Russian FastText embeddings downloaded!")
    print("\nAvailable files:")
    for file in embeddings_dir.glob("cc.ru.300.*"):
        print(f"  - {file.name} ({file.stat().st_size / (1024**3):.2f} GB)")


if __name__ == "__main__":
    main() 