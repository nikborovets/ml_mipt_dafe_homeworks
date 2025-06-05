"""
Тесты для метрик и математических функций
"""
import numpy as np
import torch
import torch.nn.functional as F
import pytest


class TestMetrics:
    """Тесты для метрик и математических операций"""
    
    def test_euclidean_distance_same_dimension(self):
        """
        Проверяет функции: вычисление евклидова расстояния между двумя векторами одинаковой размерности
        """
        # Тестовые векторы
        vector1 = torch.tensor([1.0, 2.0, 3.0])
        vector2 = torch.tensor([4.0, 5.0, 6.0])
        
        # Вычисляем евклидово расстояние
        # Расстояние между (1,2,3) и (4,5,6) = sqrt((4-1)² + (5-2)² + (6-3)²) = sqrt(9+9+9) = sqrt(27) ≈ 5.196
        distance = torch.cdist(vector1.unsqueeze(0), vector2.unsqueeze(0), p=2)
        expected_distance = torch.sqrt(torch.tensor(27.0))
        
        assert torch.allclose(distance, expected_distance, atol=1e-4), \
            f"Евклидово расстояние неверное: получено {distance.item()}, ожидалось {expected_distance.item()}"
    
    def test_euclidean_distance_zero_distance(self):
        """
        Проверяет что расстояние от вектора до самого себя равно 0
        """
        vector = torch.tensor([1.0, 2.0, 3.0])
        distance = torch.cdist(vector.unsqueeze(0), vector.unsqueeze(0), p=2)
        
        assert torch.allclose(distance, torch.tensor(0.0), atol=1e-6), \
            f"Расстояние до самого себя должно быть 0, получено {distance.item()}"
    
    def test_euclidean_distance_batch(self):
        """
        Проверяет вычисление евклидова расстояния для батча векторов
        """
        # Создаем батч из 3 векторов размерности 4
        batch1 = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0]])
        
        batch2 = torch.tensor([[0.0, 0.0, 0.0, 1.0],
                              [1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0]])
        
        # Вычисляем все попарные расстояния
        distances = torch.cdist(batch1, batch2, p=2)
        
        # Проверяем форму результата
        assert distances.shape == (3, 3), f"Форма должна быть (3, 3), получена {distances.shape}"
        
        # Проверяем что все расстояния >= 0
        assert torch.all(distances >= 0), "Все расстояния должны быть неотрицательными"
        
        # Проверяем конкретные значения
        # Расстояние между (1,0,0,0) и (0,0,0,1) = sqrt(1+1) = sqrt(2)
        expected_sqrt2 = torch.sqrt(torch.tensor(2.0))
        assert torch.allclose(distances[0, 0], expected_sqrt2, atol=1e-4), \
            f"Расстояние неверное: получено {distances[0, 0]}, ожидалось {expected_sqrt2}"
    
    def test_softmax_to_log_softmax_conversion(self):
        """
        Проверяет: softmax → log-softmax возвращает ожидаемые вероятности для малых тензоров
        """
        # Тестовый логит-вектор
        logits = torch.tensor([1.0, 2.0, 3.0])
        
        # Вычисляем softmax
        softmax_probs = F.softmax(logits, dim=0)
        
        # Вычисляем log-softmax
        log_softmax_probs = F.log_softmax(logits, dim=0)
        
        # Проверяем что log(softmax(x)) = log_softmax(x)
        log_of_softmax = torch.log(softmax_probs)
        
        assert torch.allclose(log_of_softmax, log_softmax_probs, atol=1e-6), \
            "log(softmax(x)) должно равняться log_softmax(x)"
        
        # Проверяем что softmax сумма равна 1
        assert torch.allclose(torch.sum(softmax_probs), torch.tensor(1.0), atol=1e-6), \
            f"Сумма softmax должна быть 1, получена {torch.sum(softmax_probs)}"
        
        # Проверяем что все вероятности положительные
        assert torch.all(softmax_probs > 0), "Все softmax вероятности должны быть > 0"
        
        # Проверяем что log_softmax дает отрицательные значения (кроме максимума)
        assert torch.all(log_softmax_probs <= 0), "Все log_softmax значения должны быть <= 0"
    
    def test_softmax_numerical_stability(self):
        """
        Проверяет численную стабильность softmax для больших значений
        """
        # Большие логиты
        large_logits = torch.tensor([1000.0, 1001.0, 999.0])
        
        # Должно работать без overflow
        try:
            softmax_probs = F.softmax(large_logits, dim=0)
            log_softmax_probs = F.log_softmax(large_logits, dim=0)
            
            # Проверяем что результаты конечные
            assert torch.all(torch.isfinite(softmax_probs)), "Softmax должен быть конечным"
            assert torch.all(torch.isfinite(log_softmax_probs)), "Log-softmax должен быть конечным"
            
            # Проверяем что сумма softmax равна 1
            assert torch.allclose(torch.sum(softmax_probs), torch.tensor(1.0), atol=1e-6), \
                "Сумма softmax должна быть 1 даже для больших логитов"
                
        except Exception as e:
            pytest.fail(f"Softmax должен обрабатывать большие значения: {e}")
    
    def test_distance_matrix_properties(self):
        """
        Проверяет свойства матрицы расстояний
        """
        # Создаем случайные векторы
        torch.manual_seed(42)
        vectors = torch.randn(5, 3)  # 5 векторов размерности 3
        
        # Вычисляем матрицу расстояний
        distance_matrix = torch.cdist(vectors, vectors, p=2)
        
        # Проверяем размер
        assert distance_matrix.shape == (5, 5), f"Матрица должна быть 5x5, получена {distance_matrix.shape}"
        
        # Проверяем что диагональ состоит из нулей
        diagonal = torch.diag(distance_matrix)
        assert torch.allclose(diagonal, torch.zeros(5), atol=1e-6), \
            "Диагональ матрицы расстояний должна состоять из нулей"
        
        # Проверяем симметричность
        assert torch.allclose(distance_matrix, distance_matrix.T, atol=1e-6), \
            "Матрица расстояний должна быть симметричной"
        
        # Проверяем что все расстояния неотрицательные
        assert torch.all(distance_matrix >= 0), "Все расстояния должны быть неотрицательными"
    
    def test_negative_log_likelihood_loss(self):
        """
        Проверяет вычисление Negative Log-Likelihood Loss
        """
        # Создаем log-вероятности для 3 классов
        log_probs = torch.tensor([[-1.0, -2.0, -3.0],  # Пример 1
                                 [-2.0, -1.0, -3.0]])  # Пример 2
        
        # Истинные метки
        targets = torch.tensor([0, 1])  # Пример 1 -> класс 0, Пример 2 -> класс 1
        
        # Вычисляем NLL loss
        nll_loss = F.nll_loss(log_probs, targets, reduction='mean')
        
        # Ожидаемое значение: (-(-1.0) + -(-1.0)) / 2 = (1.0 + 1.0) / 2 = 1.0
        expected_loss = torch.tensor(1.0)
        
        assert torch.allclose(nll_loss, expected_loss, atol=1e-6), \
            f"NLL loss неверный: получен {nll_loss}, ожидался {expected_loss}"
    
    def test_cross_entropy_equivalence(self):
        """
        Проверяет эквивалентность CrossEntropy и LogSoftmax + NLLLoss
        """
        torch.manual_seed(42)
        logits = torch.randn(2, 3)  # 2 примера, 3 класса
        targets = torch.tensor([0, 2])
        
        # Метод 1: CrossEntropy напрямую
        ce_loss = F.cross_entropy(logits, targets)
        
        # Метод 2: LogSoftmax + NLLLoss
        log_probs = F.log_softmax(logits, dim=1)
        nll_loss = F.nll_loss(log_probs, targets)
        
        assert torch.allclose(ce_loss, nll_loss, atol=1e-6), \
            f"CrossEntropy и LogSoftmax+NLLLoss должны быть эквивалентны: {ce_loss} vs {nll_loss}"
    
    def test_prototype_calculation(self):
        """
        Проверяет вычисление прототипов (средних) для support set
        """
        # Создаем support set: 2 класса по 3 примера каждый, векторы размерности 4
        support_embeddings = torch.tensor([
            # Класс 0
            [[1.0, 0.0, 0.0, 0.0],
             [1.1, 0.1, 0.0, 0.0],
             [0.9, -0.1, 0.0, 0.0]],
            # Класс 1
            [[0.0, 1.0, 0.0, 0.0],
             [0.0, 1.1, 0.1, 0.0],
             [0.0, 0.9, -0.1, 0.0]]
        ])  # shape: (2, 3, 4)
        
        # Вычисляем прототипы (средние по support примерам)
        prototypes = support_embeddings.mean(dim=1)  # shape: (2, 4)
        
        # Ожидаемые прототипы
        expected_proto_0 = torch.tensor([1.0, 0.0, 0.0, 0.0])  # среднее по классу 0
        expected_proto_1 = torch.tensor([0.0, 1.0, 0.0, 0.0])  # среднее по классу 1
        
        assert torch.allclose(prototypes[0], expected_proto_0, atol=1e-6), \
            f"Прототип класса 0 неверный: получен {prototypes[0]}, ожидался {expected_proto_0}"
        
        assert torch.allclose(prototypes[1], expected_proto_1, atol=1e-6), \
            f"Прототип класса 1 неверный: получен {prototypes[1]}, ожидался {expected_proto_1}"
        
        # Проверяем размерность
        assert prototypes.shape == (2, 4), f"Форма прототипов должна быть (2, 4), получена {prototypes.shape}" 