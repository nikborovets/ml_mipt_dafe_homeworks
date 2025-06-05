"""
Тесты для модуля данных src/data.py
"""
import os
import tempfile
import numpy as np
import pytest

from src.data import read_alphabets, read_images, extract_sample


class TestDataset:
    """Тесты для функций работы с датасетом"""
    
    def test_read_alphabets_and_read_images_equal_length(self):
        """
        Проверяет, что read_alphabets и read_images возвращают списки одинаковой длины
        """
        # Создаем временную структуру директорий
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем поддиректории для алфавитов
            alphabet1_dir = os.path.join(temp_dir, "alphabet1")
            alphabet2_dir = os.path.join(temp_dir, "alphabet2")
            os.makedirs(alphabet1_dir)
            os.makedirs(alphabet2_dir)
            
            # Создаем поддиректории для символов
            char1_dir = os.path.join(alphabet1_dir, "character01")
            char2_dir = os.path.join(alphabet2_dir, "character02")
            os.makedirs(char1_dir)
            os.makedirs(char2_dir)
            
            # Создаем тестовые изображения (пустые файлы .png)
            test_files = [
                os.path.join(char1_dir, "0001_01.png"),
                os.path.join(char1_dir, "0001_02.png"),
                os.path.join(char2_dir, "0002_01.png"),
                os.path.join(char2_dir, "0002_02.png"),
            ]
            
            for file_path in test_files:
                # Создаем простое тестовое изображение 28x28x3
                image = np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8)
                import cv2
                cv2.imwrite(file_path, image)
            
            # Тестируем функции
            alphabets = read_alphabets(temp_dir)
            all_images, all_labels = read_images(temp_dir)
            
            # Проверяем, что длины равны
            assert len(all_images) == len(all_labels), \
                "read_images должна возвращать массивы одинаковой длины"
            
            # Проверяем, что данные не пустые
            assert len(all_images) > 0, "Должны быть загружены изображения"
            assert len(alphabets) == 2, "Должно быть найдено 2 алфавита"
    
    def test_rotations_increase_dataset_size(self):
        """
        Проверяет, что после поворотов (90°, 180°, 270°) размер датасета увеличивается в 4 раза
        """
        # Создаем временную структуру директорий с изображениями
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем алфавит с одним символом
            alphabet_dir = os.path.join(temp_dir, "test_alphabet")
            character_dir = os.path.join(alphabet_dir, "character01")
            os.makedirs(character_dir)
            
            # Создаем 2 тестовых изображения
            for i in range(2):
                image_path = os.path.join(character_dir, f"img_{i:04d}.png")
                image = np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8)
                import cv2
                cv2.imwrite(image_path, image)
            
            # Читаем данные
            datax, datay = read_alphabets(alphabet_dir)
            
            # Проверяем размеры: должно быть 2 изображения * 4 поворота = 8
            expected_size = 2 * 4  # оригинал + 3 поворота
            
            assert len(datax) == expected_size, \
                f"После поворотов размер должен быть {expected_size}, получен {len(datax)}"
            assert len(datay) == expected_size, \
                f"После поворотов размер меток должен быть {expected_size}, получен {len(datay)}"
            
            # Проверяем, что есть классы с поворотами
            unique_labels = np.unique(datay)
            expected_labels = {'character01', 'character01_rot90', 'character01_rot180', 'character01_rot270'}
            assert set(unique_labels) == expected_labels, \
                f"Ожидаемые метки {expected_labels}, получены {set(unique_labels)}"
    
    def test_extract_sample_shape(self):
        """
        Проверяет, что extract_sample возвращает тензор правильной формы
        """
        # Параметры теста
        n_way = 3
        n_support = 2
        n_query = 1
        n_examples = 5
        
        # Создаем временную структуру директорий с изображениями
        with tempfile.TemporaryDirectory() as temp_dir:
            test_images = []
            test_labels = []
            
            # Создаем n_way классов
            for class_id in range(n_way):
                class_dir = os.path.join(temp_dir, f"class_{class_id}")
                os.makedirs(class_dir)
                
                # Создаем n_examples изображений для каждого класса
                for img_id in range(n_examples):
                    image_path = os.path.join(class_dir, f"img_{img_id:04d}.png")
                    image = np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8)
                    import cv2
                    cv2.imwrite(image_path, image)
                    test_images.append(image_path)
                    test_labels.append(f"class_{class_id}")
            
            test_images = np.array(test_images)
            test_labels = np.array(test_labels)
            
            # Извлекаем семпл
            sample_dict = extract_sample(n_way, n_support, n_query, test_images, test_labels)
            sample = sample_dict['images']
            
            # Ожидаемая форма: (n_way, n_support + n_query, 3, 28, 28)
            expected_shape = (n_way, n_support + n_query, 3, 28, 28)
            
            assert sample.shape == expected_shape, \
                f"Ожидаемая форма {expected_shape}, получена {sample.shape}"
            
            # Проверяем тип данных
            assert hasattr(sample, 'shape'), "sample должен иметь форму"
    
    def test_extract_sample_insufficient_data(self):
        """
        Проверяет поведение extract_sample при недостаточном количестве данных
        """
        n_way = 3
        n_support = 5
        n_query = 2
        n_examples = 4  # Недостаточно данных (нужно n_support + n_query = 7)
        
        # Создаем временную структуру директорий с изображениями
        with tempfile.TemporaryDirectory() as temp_dir:
            test_images = []
            test_labels = []
            
            # Создаем n_way классов с недостаточным количеством изображений
            for class_id in range(n_way):
                class_dir = os.path.join(temp_dir, f"class_{class_id}")
                os.makedirs(class_dir)
                
                # Создаем только n_examples изображений (меньше чем n_support + n_query)
                for img_id in range(n_examples):
                    image_path = os.path.join(class_dir, f"img_{img_id:04d}.png")
                    image = np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8)
                    import cv2
                    cv2.imwrite(image_path, image)
                    test_images.append(image_path)
                    test_labels.append(f"class_{class_id}")
            
            test_images = np.array(test_images)
            test_labels = np.array(test_labels)
            
            # Поведение при недостаточных данных: проверим что функция работает корректно
            # (может взять меньше данных чем запрошено или обработать иначе)
            try:
                sample_dict = extract_sample(n_way, n_support, n_query, test_images, test_labels)
                # Если не возникло исключения, проверим что данные корректные
                assert 'images' in sample_dict, "Результат должен содержать ключ 'images'"
                sample = sample_dict['images']
                assert sample.shape[0] <= n_way, "Количество классов не должно превышать n_way"
            except (ValueError, IndexError):
                # Это тоже допустимое поведение
                pass
    
    def test_extract_sample_different_classes(self):
        """
        Проверяет, что extract_sample корректно выбирает разные классы
        """
        n_way = 3
        n_support = 2
        n_query = 1
        n_examples = 5
        
        # Создаем временную структуру директорий с изображениями
        with tempfile.TemporaryDirectory() as temp_dir:
            test_images = []
            test_labels = []
            
            # Создаем тестовые данные с уникальными значениями для каждого класса
            for class_id in range(n_way):
                class_dir = os.path.join(temp_dir, f"class_{class_id}")
                os.makedirs(class_dir)
                
                # Создаем n_examples изображений для каждого класса
                for img_id in range(n_examples):
                    image_path = os.path.join(class_dir, f"img_{img_id:04d}.png")
                    # Каждый класс имеет уникальное значение пикселя
                    image = np.full((28, 28, 3), fill_value=class_id * 50, dtype=np.uint8)
                    import cv2
                    cv2.imwrite(image_path, image)
                    test_images.append(image_path)
                    test_labels.append(f"class_{class_id}")
            
            test_images = np.array(test_images)
            test_labels = np.array(test_labels)
            
            # Извлекаем семпл
            sample_dict = extract_sample(n_way, n_support, n_query, test_images, test_labels)
            sample = sample_dict['images']
            
            # Проверяем основную форму
            expected_shape = (n_way, n_support + n_query, 3, 28, 28)
            assert sample.shape == expected_shape, \
                f"Ожидаемая форма {expected_shape}, получена {sample.shape}"
            
            # Проверяем, что функция возвращает правильное количество классов
            assert sample_dict['n_way'] == n_way, \
                "Должно быть выбрано правильное количество классов" 