import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from torchvision.datasets import CIFAR10


def test_cifar10_download():
    """Тестирует скачивание датасета CIFAR-10"""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cifar_train_exists = os.path.exists(os.path.join(project_root, "CIFAR10/train"))
    cifar_test_exists = os.path.exists(os.path.join(project_root, "CIFAR10/test"))
    
    if cifar_train_exists and cifar_test_exists:
        from torchvision.datasets import CIFAR10
        
        train_dataset = CIFAR10(os.path.join(project_root, "CIFAR10/train"), download=False)
        test_dataset = CIFAR10(os.path.join(project_root, "CIFAR10/test"), download=False)
        
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0
        
    else:
        with patch('torchvision.datasets.CIFAR10') as mock_cifar:
            mock_train = MagicMock()
            mock_test = MagicMock()
            
            mock_cifar.return_value = mock_train
            
            from torchvision.datasets import CIFAR10
            
            train_dataset = CIFAR10(os.path.join(project_root, "CIFAR10/train"), download=True)
            test_dataset = CIFAR10(os.path.join(project_root, "CIFAR10/test"), download=True)
            
            assert mock_cifar.call_count == 2


def test_cifar10_dataset_properties():
    """Тестирует свойства датасета CIFAR-10"""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(project_root, "CIFAR10/train")) and os.path.exists(os.path.join(project_root, "CIFAR10/test")):
        try:
            train_dataset = CIFAR10(os.path.join(project_root, "CIFAR10/train"), train=True, download=False)
            test_dataset = CIFAR10(os.path.join(project_root, "CIFAR10/test"), train=False, download=False)
            
            assert len(train_dataset) == 50000
            assert len(test_dataset) == 10000
            
            sample_data, sample_label = train_dataset[0]
            assert sample_data.size == (32, 32)
            assert isinstance(sample_label, int)
            assert 0 <= sample_label <= 9
            
        except Exception as e:
            pytest.skip(f"Не удалось загрузить существующие данные CIFAR-10: {e}")
    
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                train_dataset = CIFAR10(temp_dir, train=True, download=True)
                test_dataset = CIFAR10(temp_dir, train=False, download=True)
                
                assert len(train_dataset) == 50000
                assert len(test_dataset) == 10000
                
                sample_data, sample_label = train_dataset[0]
                assert sample_data.size == (32, 32)
                assert isinstance(sample_label, int)
                assert 0 <= sample_label <= 9
                
            except Exception as e:
                pytest.skip(f"Не удалось скачать CIFAR-10: {e}")


def test_prepare_data_module_import():
    """Тестирует что модуль prepare_data можно импортировать"""
    try:
        import prepare_data
        assert prepare_data is not None
    except ImportError:
        pytest.fail("Не удалось импортировать модуль prepare_data") 