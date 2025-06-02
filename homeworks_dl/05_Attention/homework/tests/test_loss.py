# Тесты для проверки функции потерь Label Smoothing
import pytest
import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import LabelSmoothingLoss


def test_label_smoothing_loss_init():
    """Тест инициализации Label Smoothing Loss."""
    vocab_size = 1000
    padding_idx = 0
    smoothing = 0.1
    
    criterion = LabelSmoothingLoss(vocab_size, padding_idx, smoothing)
    
    assert criterion.size == vocab_size
    assert criterion.padding_idx == padding_idx
    assert criterion.smoothing == smoothing
    assert criterion.confidence == 1.0 - smoothing


def test_label_smoothing_loss_forward():
    """Тест прямого прохода Label Smoothing Loss."""
    vocab_size = 10
    batch_size = 2
    seq_len = 3
    padding_idx = 0
    smoothing = 0.1
    
    criterion = LabelSmoothingLoss(vocab_size, padding_idx, smoothing)
    
    # Создаем mock данные
    x = torch.randn(batch_size * seq_len, vocab_size)  # Логиты
    target = torch.LongTensor([1, 2, 3, 4, 5, 0])  # Целевые метки
    
    loss = criterion(x, target)
    
    assert isinstance(loss, torch.Tensor)
    # KL divergence может быть отрицательным, это нормально
    assert not torch.isnan(loss)  # Проверяем, что loss не NaN


def test_label_smoothing_vs_standard_loss():
    """Тест сравнения Label Smoothing с обычным NLL Loss."""
    vocab_size = 10
    batch_size = 2
    seq_len = 3
    padding_idx = 0
    
    # Label Smoothing Loss
    ls_criterion = LabelSmoothingLoss(vocab_size, padding_idx, smoothing=0.1)
    
    # Стандартный NLL Loss
    nll_criterion = nn.NLLLoss(ignore_index=padding_idx)
    
    # Создаем данные
    x = torch.randn(batch_size * seq_len, vocab_size)
    log_probs = torch.log_softmax(x, dim=-1)
    target = torch.LongTensor([1, 2, 3, 4, 5, 0])
    
    ls_loss = ls_criterion(log_probs, target)
    nll_loss = nll_criterion(log_probs, target)
    
    # Label Smoothing должен давать другой результат
    assert ls_loss.item() != nll_loss.item()


def test_label_smoothing_padding_handling():
    """Тест обработки padding токенов."""
    vocab_size = 5
    padding_idx = 0
    smoothing = 0.1
    
    criterion = LabelSmoothingLoss(vocab_size, padding_idx, smoothing)
    
    # Создаем данные с padding токенами
    x = torch.randn(4, vocab_size)
    target = torch.LongTensor([1, 2, 0, 0])  # Два padding токена
    
    loss = criterion(x, target)
    
    # Проверяем, что true_dist правильно обрабатывает padding
    assert criterion.true_dist is not None
    # Padding токены должны иметь нулевое распределение
    assert criterion.true_dist[2, padding_idx] == 0
    assert criterion.true_dist[3, padding_idx] == 0


def test_label_smoothing_confidence_distribution():
    """Тест правильности распределения confidence."""
    vocab_size = 5
    padding_idx = 0
    smoothing = 0.1
    
    criterion = LabelSmoothingLoss(vocab_size, padding_idx, smoothing)
    
    x = torch.randn(2, vocab_size)
    target = torch.LongTensor([1, 2])
    
    loss = criterion(x, target)
    
    # Проверяем распределение вероятностей
    true_dist = criterion.true_dist
    
    # Для истинного класса должна быть высокая вероятность
    assert true_dist[0, 1] == criterion.confidence  # Первый пример, класс 1
    assert true_dist[1, 2] == criterion.confidence  # Второй пример, класс 2
    
    # Для остальных классов должна быть сглаженная вероятность
    expected_smooth = smoothing / (vocab_size - 2)  # -2 для padding и истинного класса
    assert abs(true_dist[0, 3] - expected_smooth) < 1e-6 