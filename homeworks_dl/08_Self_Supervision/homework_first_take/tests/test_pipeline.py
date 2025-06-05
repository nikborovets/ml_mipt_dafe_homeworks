"""
Тесты для пайплайна обучения src/train.py
"""
import os
import tempfile
import numpy as np
import torch
import pytest

from src.data import read_images, extract_sample
from src.model import load_protonet_conv
from src.train import train, validate


class TestPipeline:
    """End-to-end тесты для пайплайна обучения"""
    
    def test_train_one_step_small_dataset(self):
        """
        End-to-end тест на искусственном малом наборе (3 класса × 5 примеров): 
        train выполняется один шаг без ошибок
        """
        # Создаем маленький искусственный датасет
        with tempfile.TemporaryDirectory() as temp_dir:
            n_classes = 3
            n_examples_per_class = 5
            
            train_x = []
            train_y = []
            
            # Создаем структуру директорий и изображения
            for class_id in range(n_classes):
                class_dir = os.path.join(temp_dir, f"class_{class_id}")
                os.makedirs(class_dir)
                
                for img_id in range(n_examples_per_class):
                    image_path = os.path.join(class_dir, f"img_{img_id:04d}.png")
                    # Создаем простые изображения с уникальными значениями для каждого класса
                    image = np.full((28, 28, 3), fill_value=class_id * 80, dtype=np.uint8)
                    import cv2
                    cv2.imwrite(image_path, image)
                    train_x.append(image_path)
                    train_y.append(f"class_{class_id}")
            
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            
            # Создаем модель
            model = load_protonet_conv(
                x_dim=(3, 28, 28),
                hid_dim=64,
                z_dim=64
            )
            
            # Создаем оптимизатор
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Параметры для одного шага обучения
            n_way = 3
            n_support = 2
            n_query = 2
            max_epoch = 1
            epoch_size = 1  # Только один эпизод
            
            # Запускаем обучение на один шаг - не должно быть ошибок
            try:
                train(
                    model=model,
                    optimizer=optimizer,
                    train_x=train_x,
                    train_y=train_y,
                    n_way=n_way,
                    n_support=n_support,
                    n_query=n_query,
                    max_epoch=max_epoch,
                    epoch_size=epoch_size,
                    writer=None,  # Без TensorBoard для теста
                    save_path=os.path.join(temp_dir, "test_model.pt")
                )
                
                # Проверяем, что модель была сохранена
                model_path = os.path.join(temp_dir, "test_model.pt")
                assert os.path.exists(model_path), "Модель должна быть сохранена"
                
                # Проверяем, что можно загрузить модель
                checkpoint = torch.load(model_path, map_location='cpu')
                assert 'model_state_dict' in checkpoint, "Checkpoint должен содержать model_state_dict"
                assert 'best_acc' in checkpoint, "Checkpoint должен содержать best_acc"
                
            except Exception as e:
                pytest.fail(f"Обучение завершилось с ошибкой: {e}")
    
    def test_pipeline_produces_valid_metrics(self):
        """
        Тест что пайплайн вычисляет метрики (accuracy ≥ 0)
        """
        # Создаем маленький искусственный датасет
        with tempfile.TemporaryDirectory() as temp_dir:
            n_classes = 2
            n_examples_per_class = 4
            
            test_x = []
            test_y = []
            
            # Создаем структуру директорий и изображения для теста
            for class_id in range(n_classes):
                class_dir = os.path.join(temp_dir, f"test_class_{class_id}")
                os.makedirs(class_dir)
                
                for img_id in range(n_examples_per_class):
                    image_path = os.path.join(class_dir, f"img_{img_id:04d}.png")
                    # Создаем изображения с отличительными характеристиками
                    image = np.full((28, 28, 3), fill_value=class_id * 100 + 50, dtype=np.uint8)
                    import cv2
                    cv2.imwrite(image_path, image)
                    test_x.append(image_path)
                    test_y.append(f"test_class_{class_id}")
            
            test_x = np.array(test_x)
            test_y = np.array(test_y)
            
            # Создаем и обучаем модель
            model = load_protonet_conv(
                x_dim=(3, 28, 28),
                hid_dim=64,
                z_dim=64
            )
            
            # Тестируем валидацию
            n_way = 2
            n_support = 1
            n_query = 1
            val_episodes = 2  # Мало эпизодов для быстроты
            
            try:
                avg_loss, avg_acc = validate(
                    model=model,
                    val_x=test_x,
                    val_y=test_y,
                    n_way=n_way,
                    n_support=n_support,
                    n_query=n_query,
                    val_episodes=val_episodes
                )
                
                # Проверяем, что метрики валидные
                assert isinstance(avg_loss, (int, float)), "Loss должен быть числом"
                assert isinstance(avg_acc, (int, float)), "Accuracy должен быть числом"
                assert avg_acc >= 0.0, f"Accuracy должен быть ≥ 0, получен {avg_acc}"
                assert avg_acc <= 1.0, f"Accuracy должен быть ≤ 1, получен {avg_acc}"
                assert avg_loss >= 0.0, f"Loss должен быть ≥ 0, получен {avg_loss}"
                
            except Exception as e:
                pytest.fail(f"Валидация завершилась с ошибкой: {e}")
    
    def test_extract_sample_integration(self):
        """
        Тест интеграции extract_sample с моделью
        """
        # Создаем маленький искусственный датасет
        with tempfile.TemporaryDirectory() as temp_dir:
            n_classes = 2
            n_examples_per_class = 3
            
            data_x = []
            data_y = []
            
            # Создаем структуру директорий и изображения
            for class_id in range(n_classes):
                class_dir = os.path.join(temp_dir, f"int_class_{class_id}")
                os.makedirs(class_dir)
                
                for img_id in range(n_examples_per_class):
                    image_path = os.path.join(class_dir, f"img_{img_id:04d}.png")
                    image = np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8)
                    import cv2
                    cv2.imwrite(image_path, image)
                    data_x.append(image_path)
                    data_y.append(f"int_class_{class_id}")
            
            data_x = np.array(data_x)
            data_y = np.array(data_y)
            
            # Создаем модель
            model = load_protonet_conv(
                x_dim=(3, 28, 28),
                hid_dim=64,
                z_dim=64
            )
            
            # Извлекаем семпл
            n_way = 2
            n_support = 1
            n_query = 1
            
            sample = extract_sample(n_way, n_support, n_query, data_x, data_y)
            
            # Проверяем что семпл можно передать в модель
            try:
                loss, output = model.set_forward_loss(sample)
                
                # Проверяем что модель возвращает корректные результаты
                assert isinstance(loss, torch.Tensor), "Loss должен быть тензором"
                assert isinstance(output, dict), "Output должен быть словарем"
                assert 'loss' in output, "Output должен содержать ключ 'loss'"
                assert 'acc' in output, "Output должен содержать ключ 'acc'"
                
                # Проверяем значения
                assert output['acc'] >= 0.0, f"Accuracy должен быть ≥ 0, получен {output['acc']}"
                assert output['acc'] <= 1.0, f"Accuracy должен быть ≤ 1, получен {output['acc']}"
                assert output['loss'] >= 0.0, f"Loss должен быть ≥ 0, получен {output['loss']}"
                
            except Exception as e:
                pytest.fail(f"Модель не смогла обработать семпл: {e}")
    
    def test_model_training_reduces_loss(self):
        """
        Тест что несколько шагов обучения приводят к уменьшению loss
        """
        # Создаем маленький но предсказуемый датасет
        with tempfile.TemporaryDirectory() as temp_dir:
            n_classes = 2
            n_examples_per_class = 6
            
            train_x = []
            train_y = []
            
            # Создаем очень простые и отличительные изображения
            for class_id in range(n_classes):
                class_dir = os.path.join(temp_dir, f"simple_class_{class_id}")
                os.makedirs(class_dir)
                
                for img_id in range(n_examples_per_class):
                    image_path = os.path.join(class_dir, f"img_{img_id:04d}.png")
                    # Класс 0: черные изображения, Класс 1: белые изображения
                    pixel_value = 0 if class_id == 0 else 255
                    image = np.full((28, 28, 3), fill_value=pixel_value, dtype=np.uint8)
                    import cv2
                    cv2.imwrite(image_path, image)
                    train_x.append(image_path)
                    train_y.append(f"simple_class_{class_id}")
            
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            
            # Создаем модель
            model = load_protonet_conv(
                x_dim=(3, 28, 28),
                hid_dim=64,
                z_dim=64
            )
            
            # Измеряем начальный loss
            sample = extract_sample(2, 2, 2, train_x, train_y, seed=42)
            initial_loss, initial_output = model.set_forward_loss(sample)
            initial_loss_value = initial_output['loss']
            
            # Обучаем модель
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Более высокий LR
            
            # Несколько шагов обучения на том же семпле
            for _ in range(5):
                optimizer.zero_grad()
                loss, output = model.set_forward_loss(sample)
                loss.backward()
                optimizer.step()
            
            # Измеряем финальный loss
            with torch.no_grad():
                final_loss, final_output = model.set_forward_loss(sample)
                final_loss_value = final_output['loss']
            
            # Проверяем что loss уменьшился (или хотя бы не сильно увеличился)
            assert final_loss_value <= initial_loss_value * 1.1, \
                f"Loss должен уменьшиться или остаться примерно тем же. " \
                f"Начальный: {initial_loss_value:.4f}, Финальный: {final_loss_value:.4f}" 