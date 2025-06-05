"""
Модуль для работы с данными Omniglot.
Включает функции загрузки данных, создания эпизодов и аугментации.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


def read_alphabets(alphabet_directory_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    # from omniglot_hw_from_ipynb.py: читает все символы из заданного алфавита
    Reads all the characters from a given alphabet_directory
    Args:
      alphabet_directory_path (str): path to directory with files
    Returns:
      datax (np.array): array of path name of images
      datay (np.array): array of labels
    """
    datax = []  # all file names of images
    datay = []  # all class names 
    
    # Проходим по всем символам в алфавите
    for character_dir in os.listdir(alphabet_directory_path):
        character_path = os.path.join(alphabet_directory_path, character_dir)
        if os.path.isdir(character_path):
            # Читаем все изображения символа
            for image_file in os.listdir(character_path):
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(character_path, image_file)
                    datax.append(image_path)
                    datay.append(character_dir)
                    
                    # Добавляем повернутые версии (90°, 180°, 270°)
                    for angle in [90, 180, 270]:
                        datax.append(image_path)
                        datay.append(f"{character_dir}_rot{angle}")
    
    return np.array(datax), np.array(datay)


def read_images(base_directory: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    # from omniglot_hw_from_ipynb.py: читает все алфавиты из base_directory
    Reads all the alphabets from the base_directory
    Uses multithreading to decrease the reading time drastically
    """
    datax = None
    datay = None
    
    results = [read_alphabets(base_directory + '/' + directory + '/') 
               for directory in os.listdir(base_directory) 
               if os.path.isdir(os.path.join(base_directory, directory))]
    
    for result in results:
        if datax is None:
            datax = result[0]
            datay = result[1]
        else:
            datax = np.concatenate([datax, result[0]])
            datay = np.concatenate([datay, result[1]])
            
    return datax, datay


def get_augmentation_transform(seed: Optional[int] = None) -> A.Compose:
    """
    Создает композицию аугментаций для изображений Omniglot
    Args:
        seed: случайное зерно для детерминированности
    Returns:
        albumentations.Compose: композиция трансформаций
    """
    if seed is not None:
        A.core.transforms_interface.BasicTransform.__init__ = lambda self, *args, **kwargs: None
        
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return transform


def apply_rotation(image: np.ndarray, rotation_angle: int) -> np.ndarray:
    """
    Применяет поворот к изображению
    Args:
        image: исходное изображение
        rotation_angle: угол поворота (90, 180, 270)
    Returns:
        повернутое изображение
    """
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image


def extract_sample(n_way: int, n_support: int, n_query: int, 
                  datax: np.ndarray, datay: np.ndarray, 
                  seed: Optional[int] = None) -> Dict:
    """
    # from omniglot_hw_from_ipynb.py: создает случайную выборку для эпизода
    Picks random sample of size n_support + n_query, for n_way classes
    Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (np.array): dataset of images
      datay (np.array): dataset of labels
      seed (int): random seed for reproducibility
    Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support + n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
    """
    if seed is not None:
        np.random.seed(seed)
        
    sample = []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    
    # Получаем трансформацию для аугментации
    transform = get_augmentation_transform(seed)
    
    for cls in K:
        datax_cls = datax[datay == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support + n_query)]
        
        processed_images = []
        for fname in sample_cls:
            # Загружаем изображение
            image = cv2.imread(fname)
            image = cv2.resize(image, (28, 28))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Применяем поворот если нужно
            if '_rot' in cls:
                angle = int(cls.split('_rot')[1])
                image = apply_rotation(image, angle)
            
            # Применяем аугментации
            augmented = transform(image=image)
            processed_images.append(augmented['image'])
            
        sample.append(torch.stack(processed_images))
        
    sample = torch.stack(sample)
    
    return {
        'images': sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    }


def display_sample(sample: torch.Tensor) -> None:
    """
    # from omniglot_hw_from_ipynb.py: отображает выборку в виде сетки
    Displays sample in a grid
    Args:
      sample (torch.Tensor): sample of images to display
    """
    # need 4D tensor to create grid, currently 5D
    sample_4D = sample.view(sample.shape[0] * sample.shape[1], *sample.shape[2:])
    # make a grid
    out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])
    
    plt.figure(figsize=(16, 7))
    plt.imshow(out.permute(1, 2, 0))
    plt.axis('off')
    plt.tight_layout()


def prepare_data_for_training(base_dir: str = ".", download: bool = True) -> None:
    """
    Подготавливает данные для обучения - загружает датасет Omniglot
    Args:
        base_dir: базовая директория для сохранения данных
        download: флаг загрузки данных
    """
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    if download:
        import urllib.request
        import zipfile
        
        # URLs для датасета Omniglot
        background_url = "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip"
        evaluation_url = "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip"
        
        # Загружаем файлы если их нет
        for url, filename in [(background_url, "images_background.zip"), 
                             (evaluation_url, "images_evaluation.zip")]:
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                
                # Распаковываем
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print(f"Extracted {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data preparation for Omniglot')
    parser.add_argument('--prepare', action='store_true', help='Download and prepare data')
    parser.add_argument('--test', action='store_true', help='Test data loading')
    
    args = parser.parse_args()
    
    if args.prepare:
        prepare_data_for_training()
        print("Data preparation completed!")
        
    if args.test:
        # Тестируем загрузку данных
        try:
            trainx, trainy = read_images('data/images_background')
            testx, testy = read_images('data/images_evaluation')
            print(f"Train data: {trainx.shape}, {trainy.shape}")
            print(f"Test data: {testx.shape}, {testy.shape}")
            
            # Тестируем создание эпизода
            sample = extract_sample(5, 5, 5, trainx, trainy)
            print(f"Sample shape: {sample['images'].shape}")
            
        except Exception as e:
            print(f"Error during data test: {e}") 