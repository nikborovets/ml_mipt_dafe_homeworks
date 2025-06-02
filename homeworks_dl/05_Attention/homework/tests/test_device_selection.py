# Тест для проверки логики выбора устройства
import pytest
import torch
from unittest.mock import patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import get_device


def test_device_selection_mps_available():
    """Тест выбора MPS когда доступен."""
    with patch('torch.backends.mps.is_available', return_value=True), \
         patch('torch.cuda.is_available', return_value=False):
        device = get_device()
        assert device.type == 'mps'


def test_device_selection_cuda_available():
    """Тест выбора CUDA когда MPS недоступен, но CUDA доступен."""
    with patch('torch.backends.mps.is_available', return_value=False), \
         patch('torch.cuda.is_available', return_value=True):
        device = get_device()
        assert device.type == 'cuda'


def test_device_selection_cpu_fallback():
    """Тест выбора CPU когда ни MPS, ни CUDA недоступны."""
    with patch('torch.backends.mps.is_available', return_value=False), \
         patch('torch.cuda.is_available', return_value=False):
        device = get_device()
        assert device.type == 'cpu'


def test_device_selection_priority():
    """Тест приоритета: MPS > CUDA > CPU."""
    # Когда все доступны, должен выбрать MPS
    with patch('torch.backends.mps.is_available', return_value=True), \
         patch('torch.cuda.is_available', return_value=True):
        device = get_device()
        assert device.type == 'mps'


def test_device_returns_torch_device():
    """Тест что функция возвращает объект torch.device."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ['mps', 'cuda', 'cpu'] 