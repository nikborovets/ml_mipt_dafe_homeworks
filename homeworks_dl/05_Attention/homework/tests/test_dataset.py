# Тесты для проверки датасета
import pytest
import torch
import pandas as pd
from torchtext.data import Field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_and_process_data, convert_batch, make_mask, subsequent_mask


def test_load_and_process_data():
    """Тест загрузки и обработки данных."""
    # Проверяем, что функция возвращает правильные типы
    train_iter, test_iter, word_field = load_and_process_data(
        csv_path='news.csv', train_ratio=0.85, min_freq=7, device='cpu'
    )
    
    assert train_iter is not None
    assert test_iter is not None
    assert word_field is not None
    assert hasattr(word_field, 'vocab')
    assert len(word_field.vocab) > 0


def test_convert_batch():
    """Тест конвертации батча."""
    # Создаем mock batch
    class MockBatch:
        def __init__(self):
            self.source = torch.LongTensor([[1, 2, 3, 0], [4, 5, 0, 0]]).transpose(0, 1)
            self.target = torch.LongTensor([[1, 2, 3, 0], [4, 5, 6, 0]]).transpose(0, 1)
    
    batch = MockBatch()
    source_inputs, target_inputs, target_outputs, source_mask, target_mask = convert_batch(batch, pad_idx=0)
    
    assert source_inputs.shape[1] == 3  # Убираем последний токен
    assert target_inputs.shape[1] == 3  # Убираем последний токен
    assert target_outputs.shape[1] == 3  # Сдвигаем на 1
    assert source_mask is not None
    assert target_mask is not None


def test_make_mask():
    """Тест создания масок."""
    source_inputs = torch.LongTensor([[1, 2, 3], [4, 5, 0]])
    target_inputs = torch.LongTensor([[1, 2, 3], [4, 0, 0]])
    
    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx=0)
    
    assert source_mask.shape == (2, 1, 3)
    assert target_mask.shape == (2, 3, 3)  # Треугольная маска из-за subsequent_mask


def test_subsequent_mask():
    """Тест маски для предотвращения просмотра будущих токенов."""
    mask = subsequent_mask(4)
    
    assert mask.shape == (1, 4, 4)
    # Проверяем, что маска треугольная
    assert mask[0, 0, 1] == False  # Не можем смотреть в будущее
    assert mask[0, 1, 0] == True   # Можем смотреть в прошлое


def test_vocabulary_tokens():
    """Тест наличия специальных токенов в словаре."""
    _, _, word_field = load_and_process_data(csv_path='news.csv', device='cpu')
    
    vocab = word_field.vocab
    assert '<s>' in vocab.stoi
    assert '</s>' in vocab.stoi
    assert '<pad>' in vocab.stoi
    assert '<unk>' in vocab.stoi 