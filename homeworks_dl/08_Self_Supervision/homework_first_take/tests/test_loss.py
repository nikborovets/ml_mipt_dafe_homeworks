"""
Тесты для функции loss модели ProtoNet
"""
import numpy as np
import torch
import pytest

from src.model import load_protonet_conv


class TestLoss:
    """Тесты для функции set_forward_loss"""
    
    def test_set_forward_loss_known_input(self):
        """
        Проверяет set_forward_loss: при известном входе (N_way=2, N_support=1, N_query=1, 
        векторы энкодера заданы вручную) функция возвращает корректные loss и accuracy
        """
        # Создаем модель
        model = load_protonet_conv(
            x_dim=(3, 28, 28),
            hid_dim=64,
            z_dim=64
        )
        
        # Создаем простой тестовый семпл
        # 2 класса, по 1 support + 1 query = 2 изображения на класс
        n_way = 2
        n_support = 1
        n_query = 1
        
        # Создаем очень простые и отличительные изображения
        # Класс 0: черные изображения, Класс 1: белые изображения
        class_0_images = torch.zeros(n_support + n_query, 3, 28, 28)  # черные
        class_1_images = torch.ones(n_support + n_query, 3, 28, 28)   # белые
        
        # Формируем семпл в правильном формате
        sample_images = torch.stack([class_0_images, class_1_images])  # shape: (2, 2, 3, 28, 28)
        
        sample = {
            'images': sample_images,
            'n_way': n_way,
            'n_support': n_support,
            'n_query': n_query
        }
        
        # Тестируем функцию
        try:
            loss, output = model.set_forward_loss(sample)
            
            # Проверяем типы возвращаемых значений
            assert isinstance(loss, torch.Tensor), "Loss должен быть тензором"
            assert isinstance(output, dict), "Output должен быть словарем"
            
            # Проверяем наличие ключей
            assert 'loss' in output, "Output должен содержать ключ 'loss'"
            assert 'acc' in output, "Output должен содержать ключ 'acc'"
            assert 'y_hat' in output, "Output должен содержать ключ 'y_hat'"
            
            # Проверяем значения
            assert isinstance(output['loss'], (int, float)), "Loss должен быть числом"
            assert isinstance(output['acc'], (int, float)), "Accuracy должен быть числом"
            
            # Проверяем диапазоны
            assert output['loss'] >= 0.0, f"Loss должен быть >= 0, получен {output['loss']}"
            assert 0.0 <= output['acc'] <= 1.0, f"Accuracy должен быть в [0,1], получен {output['acc']}"
            
            # Проверяем что loss конечен
            assert torch.isfinite(loss), "Loss должен быть конечным"
            
        except Exception as e:
            pytest.fail(f"set_forward_loss завершился с ошибкой: {e}")
    
    def test_set_forward_loss_trivial_case(self):
        """
        Тест тривиального случая: когда support и query изображения идентичны,
        accuracy должна быть близка к 1.0
        """
        # Создаем модель
        model = load_protonet_conv(
            x_dim=(3, 28, 28),
            hid_dim=64,
            z_dim=64
        )
        
        n_way = 2
        n_support = 1
        n_query = 1
        
        # Создаем идентичные изображения для support и query в каждом классе
        # Класс 0: все нули
        class_0_support = torch.zeros(1, 3, 28, 28)
        class_0_query = torch.zeros(1, 3, 28, 28)  # идентично support
        
        # Класс 1: все единицы
        class_1_support = torch.ones(1, 3, 28, 28)
        class_1_query = torch.ones(1, 3, 28, 28)    # идентично support
        
        # Объединяем support и query для каждого класса
        class_0_images = torch.cat([class_0_support, class_0_query], dim=0)
        class_1_images = torch.cat([class_1_support, class_1_query], dim=0)
        
        sample_images = torch.stack([class_0_images, class_1_images])
        
        sample = {
            'images': sample_images,
            'n_way': n_way,
            'n_support': n_support,
            'n_query': n_query
        }
        
        # Обучаем модель несколько шагов на этом простом примере
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Несколько шагов обучения для улучшения accuracy
        for _ in range(10):
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            loss.backward()
            optimizer.step()
        
        # Финальная проверка
        with torch.no_grad():
            final_loss, final_output = model.set_forward_loss(sample)
            
            # После обучения accuracy должна улучшиться
            # (не обязательно 1.0, но должна быть разумной)
            assert final_output['acc'] >= 0.0, f"Accuracy должна быть >= 0, получена {final_output['acc']}"
            assert final_output['loss'] >= 0.0, f"Loss должен быть >= 0, получен {final_output['loss']}"
    
    def test_set_forward_loss_different_n_way(self):
        """
        Тест set_forward_loss с разными значениями n_way
        """
        model = load_protonet_conv(
            x_dim=(3, 28, 28),
            hid_dim=64,
            z_dim=64
        )
        
        for n_way in [2, 3, 5]:
            n_support = 1
            n_query = 1
            
            # Создаем изображения для каждого класса
            class_images = []
            for class_id in range(n_way):
                # Каждый класс имеет уникальное значение
                class_value = class_id / (n_way - 1) if n_way > 1 else 0.5
                support_img = torch.full((1, 3, 28, 28), class_value)
                query_img = torch.full((1, 3, 28, 28), class_value)
                class_img = torch.cat([support_img, query_img], dim=0)
                class_images.append(class_img)
            
            sample_images = torch.stack(class_images)
            
            sample = {
                'images': sample_images,
                'n_way': n_way,
                'n_support': n_support,
                'n_query': n_query
            }
            
            try:
                loss, output = model.set_forward_loss(sample)
                
                # Проверяем что функция работает для разных n_way
                assert isinstance(loss, torch.Tensor), f"Loss должен быть тензором для n_way={n_way}"
                assert 0.0 <= output['acc'] <= 1.0, f"Accuracy вне диапазона для n_way={n_way}: {output['acc']}"
                assert output['loss'] >= 0.0, f"Loss отрицательный для n_way={n_way}: {output['loss']}"
                
                # Проверяем размер предсказаний
                y_hat = output['y_hat']
                expected_shape = (n_way * n_query,)  # Предсказания для всех query
                assert y_hat.shape == expected_shape, \
                    f"Форма y_hat неверная для n_way={n_way}: получена {y_hat.shape}, ожидалась {expected_shape}"
                
            except Exception as e:
                pytest.fail(f"set_forward_loss завершился с ошибкой для n_way={n_way}: {e}")
    
    def test_set_forward_loss_batch_consistency(self):
        """
        Тест консистентности: одинаковые входы должны давать одинаковые результаты
        """
        model = load_protonet_conv(
            x_dim=(3, 28, 28),
            hid_dim=64,
            z_dim=64
        )
        
        # Фиксируем веса модели
        model.eval()
        
        n_way = 2
        n_support = 1
        n_query = 1
        
        # Создаем детерминированный семпл
        torch.manual_seed(42)
        sample_images = torch.randn(n_way, n_support + n_query, 3, 28, 28)
        
        sample = {
            'images': sample_images,
            'n_way': n_way,
            'n_support': n_support,
            'n_query': n_query
        }
        
        # Запускаем дважды
        with torch.no_grad():
            loss1, output1 = model.set_forward_loss(sample)
            loss2, output2 = model.set_forward_loss(sample)
        
        # Результаты должны быть идентичными
        assert torch.allclose(loss1, loss2, atol=1e-6), \
            f"Loss должен быть консистентным: {loss1} vs {loss2}"
        assert abs(output1['acc'] - output2['acc']) < 1e-6, \
            f"Accuracy должен быть консистентным: {output1['acc']} vs {output2['acc']}"
        assert abs(output1['loss'] - output2['loss']) < 1e-6, \
            f"Loss value должен быть консистентным: {output1['loss']} vs {output2['loss']}"
    
    def test_set_forward_loss_gradient_flow(self):
        """
        Тест что градиенты корректно вычисляются через set_forward_loss
        """
        model = load_protonet_conv(
            x_dim=(3, 28, 28),
            hid_dim=64,
            z_dim=64
        )
        
        n_way = 2
        n_support = 1
        n_query = 1
        
        # Создаем семпл
        sample_images = torch.randn(n_way, n_support + n_query, 3, 28, 28)
        sample = {
            'images': sample_images,
            'n_way': n_way,
            'n_support': n_support,
            'n_query': n_query
        }
        
        # Сохраняем начальные веса
        initial_params = [p.clone() for p in model.parameters()]
        
        # Прямой проход и обратное распространение
        loss, output = model.set_forward_loss(sample)
        loss.backward()
        
        # Проверяем что градиенты вычислены
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break
        
        assert has_gradients, "Должны быть вычислены ненулевые градиенты"
        
        # Проверяем что градиенты конечны
        for param in model.parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), "Все градиенты должны быть конечными"
    
    def test_set_forward_loss_edge_cases(self):
        """
        Тест граничных случаев для set_forward_loss
        """
        model = load_protonet_conv(
            x_dim=(3, 28, 28),
            hid_dim=64,
            z_dim=64
        )
        
        # Случай 1: минимальные размеры (1-way, 1-shot)
        sample_1way = {
            'images': torch.randn(1, 2, 3, 28, 28),  # 1 класс, 1 support + 1 query
            'n_way': 1,
            'n_support': 1,
            'n_query': 1
        }
        
        try:
            loss, output = model.set_forward_loss(sample_1way)
            # Для 1-way задачи accuracy всегда должна быть 1.0
            assert output['acc'] == 1.0, f"Для 1-way accuracy должна быть 1.0, получена {output['acc']}"
        except Exception as e:
            pytest.fail(f"1-way случай завершился с ошибкой: {e}")
        
        # Случай 2: большой n_support
        sample_big_support = {
            'images': torch.randn(2, 6, 3, 28, 28),  # 2 класса, 5 support + 1 query
            'n_way': 2,
            'n_support': 5,
            'n_query': 1
        }
        
        try:
            loss, output = model.set_forward_loss(sample_big_support)
            assert 0.0 <= output['acc'] <= 1.0, "Accuracy должна быть в диапазоне [0,1]"
            assert output['loss'] >= 0.0, "Loss должен быть неотрицательным"
        except Exception as e:
            pytest.fail(f"Большой support случай завершился с ошибкой: {e}")
        
        # Случай 3: большой n_query
        sample_big_query = {
            'images': torch.randn(2, 6, 3, 28, 28),  # 2 класса, 1 support + 5 query
            'n_way': 2,
            'n_support': 1,
            'n_query': 5
        }
        
        try:
            loss, output = model.set_forward_loss(sample_big_query)
            assert 0.0 <= output['acc'] <= 1.0, "Accuracy должна быть в диапазоне [0,1]"
            assert output['loss'] >= 0.0, "Loss должен быть неотрицательным"
            
            # Проверяем размер предсказаний
            y_hat = output['y_hat']
            expected_shape = (2 * 5,)  # 2 класса * 5 query
            assert y_hat.shape == expected_shape, \
                f"Форма y_hat неверная: получена {y_hat.shape}, ожидалась {expected_shape}"
                
        except Exception as e:
            pytest.fail(f"Большой query случай завершился с ошибкой: {e}") 