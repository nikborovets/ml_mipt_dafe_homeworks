import pytest
import torch
import json
import tempfile
import os
from unittest.mock import patch, mock_open, MagicMock
from torch.utils.data import TensorDataset, DataLoader

from compute_metrics import main
from train import create_model


@pytest.fixture
def mock_test_dataset():
    """Создает мок тестового датасета"""
    num_samples = 50
    test_data = torch.randn(num_samples, 3, 32, 32)
    test_labels = torch.randint(0, 10, (num_samples,))
    return TensorDataset(test_data, test_labels)


@pytest.fixture
def saved_model():
    """Создает и сохраняет модель для тестирования"""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        model_path = tmp_file.name
    
    try:
        model = create_model()
        torch.save(model.state_dict(), model_path)
        yield model_path
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_compute_metrics_main(mock_test_dataset, saved_model):
    """Тестирует основную функцию compute_metrics"""
    
    with patch('compute_metrics.CIFAR10') as mock_cifar:
        mock_cifar.return_value = mock_test_dataset
        
        with patch('compute_metrics.torch.load') as mock_load:
            model = create_model(device=torch.device('cpu'))
            mock_load.return_value = model.state_dict()
            
            with patch('compute_metrics.torch.device', return_value=torch.device('cpu')):
                
                with patch('builtins.open', mock_open()) as mock_file:
                    with patch('json.dump') as mock_json_dump:
                        
                        # Создаём mock аргументы
                        class MockArgs:
                            pass
                        
                        args = MockArgs()
                        
                        main(args)
                        
                        mock_json_dump.assert_called_once()
                        
                        call_args = mock_json_dump.call_args[0]
                        metrics_dict = call_args[0]
                        
                        assert 'accuracy' in metrics_dict
                        assert isinstance(metrics_dict['accuracy'], float)
                        assert 0.0 <= metrics_dict['accuracy'] <= 1.0


def test_compute_metrics_with_device_handling():
    """Тестирует обработку различных устройств в compute_metrics"""
    
    test_data = torch.randn(10, 3, 32, 32)
    test_labels = torch.randint(0, 10, (10,))
    mock_dataset = TensorDataset(test_data, test_labels)
    
    with patch('compute_metrics.CIFAR10') as mock_cifar:
        mock_cifar.return_value = mock_dataset
        
        with patch('compute_metrics.torch.load') as mock_load:
            model = create_model(device=torch.device('cpu'))
            mock_load.return_value = model.state_dict()
            

            with patch('compute_metrics.torch.device', return_value=torch.device('cpu')):
                with patch('builtins.open', mock_open()):
                    with patch('json.dump') as mock_json_dump:
                        
                        class MockArgs:
                            pass
                        
                        main(MockArgs())
                        
                        assert mock_json_dump.called


def test_compute_metrics_mps_device():
    """Тестирует выбор MPS устройства когда доступно"""
    test_data = torch.randn(10, 3, 32, 32)
    test_labels = torch.randint(0, 10, (10,))
    mock_dataset = TensorDataset(test_data, test_labels)
    
    with patch('compute_metrics.CIFAR10', return_value=mock_dataset):
        with patch('compute_metrics.torch.load') as mock_load:
            model = create_model(device=torch.device('cpu'))
            mock_load.return_value = model.state_dict()
            
            with patch('compute_metrics.torch.backends.mps.is_available', return_value=True):
                with patch('builtins.open', mock_open()):
                    with patch('json.dump'):
                        main(type('MockArgs', (), {})())


def test_compute_metrics_cuda_device():
    """Тестирует выбор CUDA устройства когда доступно но не MPS"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA не доступна в системе")
        
    test_data = torch.randn(10, 3, 32, 32)
    test_labels = torch.randint(0, 10, (10,))
    mock_dataset = TensorDataset(test_data, test_labels)
    
    with patch('compute_metrics.CIFAR10', return_value=mock_dataset):
        with patch('compute_metrics.torch.load') as mock_load:
            model = create_model(device=torch.device('cpu'))
            mock_load.return_value = model.state_dict()
            
            with patch('compute_metrics.torch.backends.mps.is_available', return_value=False):
                with patch('compute_metrics.torch.cuda.is_available', return_value=True):
                    with patch('builtins.open', mock_open()):
                        with patch('json.dump'):
                            main(type('MockArgs', (), {})())


def test_compute_metrics_cpu_device():
    """Тестирует выбор CPU устройства когда нет GPU"""
    test_data = torch.randn(10, 3, 32, 32)
    test_labels = torch.randint(0, 10, (10,))
    mock_dataset = TensorDataset(test_data, test_labels)
    
    with patch('compute_metrics.CIFAR10', return_value=mock_dataset):
        with patch('compute_metrics.torch.load') as mock_load:
            model = create_model(device=torch.device('cpu'))
            mock_load.return_value = model.state_dict()
            
            with patch('compute_metrics.torch.backends.mps.is_available', return_value=False):
                with patch('compute_metrics.torch.cuda.is_available', return_value=False):
                    with patch('builtins.open', mock_open()):
                        with patch('json.dump'):
                            main(type('MockArgs', (), {})())


def test_compute_metrics_file_print():
    """Тестирует print в файл (строка 58-60)"""
    test_data = torch.randn(10, 3, 32, 32)
    test_labels = torch.randint(0, 10, (10,))
    mock_dataset = TensorDataset(test_data, test_labels)
    
    with patch('compute_metrics.CIFAR10', return_value=mock_dataset):
        with patch('compute_metrics.torch.load') as mock_load:
            model = create_model(device=torch.device('cpu'))
            mock_load.return_value = model.state_dict()
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                with patch('compute_metrics.open', create=True) as mock_open_func:
                    import builtins
                    real_open = builtins.open
                    
                    def side_effect(filename, mode='r', **kwargs):
                        if filename == "final_metrics.json":
                            return real_open(tmp_path, mode, **kwargs)
                        return real_open(filename, mode, **kwargs)
                    
                    mock_open_func.side_effect = side_effect
                    
                    main(type('MockArgs', (), {})())
                    
                    assert os.path.exists(tmp_path)
                    
                    with open(tmp_path, 'r') as f:
                        content = f.read()
                        assert 'accuracy' in content
                        assert '\n' in content
                        
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path) 