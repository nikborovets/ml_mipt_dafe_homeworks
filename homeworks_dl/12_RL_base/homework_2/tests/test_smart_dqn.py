import pytest
import torch
import numpy as np
import sys
import os

# Добавляем корневую папку проекта в Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_dqn_agent import SmartDQN, SmartAgent


class TestSmartDQN:
    """Тесты для нейронной сети SmartDQN"""
    
    def test_smartdqn_initialization(self):
        """Тест инициализации SmartDQN с параметрами по умолчанию"""
        model = SmartDQN()
        assert model.fc1.in_features == 5
        assert model.fc1.out_features == 256
        assert model.fc4.out_features == 2
        
    def test_smartdqn_custom_parameters(self):
        """Тест инициализации SmartDQN с кастомными параметрами"""
        model = SmartDQN(input_size=10, hidden_size=128, output_size=3)
        assert model.fc1.in_features == 10
        assert model.fc1.out_features == 128
        assert model.fc4.out_features == 3
        
    def test_forward_pass_single_input(self):
        """Тест forward pass с одним входом"""
        model = SmartDQN()
        model.eval()
        
        # Тест с одним состоянием (5 признаков)
        input_tensor = torch.randn(5)
        output = model(input_tensor)
        
        assert output.shape == (1, 2), f"Ожидался размер (1, 2), получен {output.shape}"
        assert torch.is_tensor(output)
        
    def test_forward_pass_batch_input(self):
        """Тест forward pass с батчем входов"""
        model = SmartDQN()
        model.eval()
        
        # Тест с батчем состояний
        batch_size = 32
        input_tensor = torch.randn(batch_size, 5)
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 2), f"Ожидался размер ({batch_size}, 2), получен {output.shape}"
        assert torch.is_tensor(output)
        
    def test_forward_pass_values_range(self):
        """Тест, что выходные значения имеют разумный диапазон"""
        model = SmartDQN()
        model.eval()
        
        input_tensor = torch.randn(10, 5)
        output = model(input_tensor)
        
        # Q-значения не должны быть экстремально большими
        assert torch.all(torch.abs(output) < 1000), "Q-значения слишком большие"
        
    def test_gradient_flow(self):
        """Тест, что градиенты корректно проходят через сеть"""
        model = SmartDQN()
        model.train()
        
        # Используем батч размером 2, так как BatchNorm требует больше одного элемента
        input_tensor = torch.randn(2, 5, requires_grad=True)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Проверяем, что градиенты не None
        for param in model.parameters():
            assert param.grad is not None, "Градиенты не вычислены"
            
    def test_model_parameters_count(self):
        """Тест количества параметров модели"""
        model = SmartDQN()
        param_count = sum(p.numel() for p in model.parameters())
        
        # Реальное количество параметров для архитектуры (5->256->256->128->2)
        # включая BatchNorm параметры
        expected_min = 90000   # Минимальное ожидаемое количество
        expected_max = 120000  # Максимальное ожидаемое количество
        
        assert expected_min <= param_count <= expected_max, \
            f"Количество параметров {param_count} вне ожидаемого диапазона [{expected_min}, {expected_max}]"
            
    def test_batch_normalization_layers(self):
        """Тест наличия BatchNorm слоев"""
        model = SmartDQN()
        
        # Проверяем наличие BatchNorm слоев
        has_bn1 = hasattr(model, 'bn1') and isinstance(model.bn1, torch.nn.BatchNorm1d)
        has_bn2 = hasattr(model, 'bn2') and isinstance(model.bn2, torch.nn.BatchNorm1d)
        
        assert has_bn1, "BatchNorm1d слой bn1 отсутствует"
        assert has_bn2, "BatchNorm1d слой bn2 отсутствует"
        
    def test_dropout_layers(self):
        """Тест наличия Dropout слоев"""
        model = SmartDQN()
        
        # Проверяем наличие Dropout слоев
        has_dropout1 = hasattr(model, 'dropout1') and isinstance(model.dropout1, torch.nn.Dropout)
        has_dropout2 = hasattr(model, 'dropout2') and isinstance(model.dropout2, torch.nn.Dropout)
        
        assert has_dropout1, "Dropout слой dropout1 отсутствует"
        assert has_dropout2, "Dropout слой dropout2 отсутствует"
        
    def test_different_batch_sizes(self):
        """Тест работы с разными размерами батчей"""
        model = SmartDQN()
        model.eval()
        
        batch_sizes = [1, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 5)
            output = model(input_tensor)
            assert output.shape == (batch_size, 2), \
                f"Неправильный размер выхода для batch_size={batch_size}" 