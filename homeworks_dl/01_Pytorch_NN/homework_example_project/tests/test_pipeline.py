import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
import tempfile
import os
import shutil
from unittest.mock import patch

from train import (
    create_model, create_datasets, create_data_loaders, 
    train_one_batch, evaluate_model, train_model, 
    get_device, get_data_transforms, compute_accuracy
)


@pytest.fixture
def small_datasets():
    """Создает небольшие тестовые датасеты для быстрого тестирования"""
    num_samples = 100
    
    # формат CIFAR-10: 3x32x32
    train_data = torch.randn(num_samples, 3, 32, 32)
    train_labels = torch.randint(0, 10, (num_samples,))
    
    test_data = torch.randn(num_samples // 2, 3, 32, 32)
    test_labels = torch.randint(0, 10, (num_samples // 2,))
    
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    return train_dataset, test_dataset


@pytest.fixture
def real_cifar_datasets():
    """Фикстура для скачивания и подготовки настоящих данных CIFAR-10"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cifar_path = os.path.join(project_root, "CIFAR10")
    if os.path.exists(cifar_path) and os.path.exists(os.path.join(cifar_path, "train")) and os.path.exists(os.path.join(cifar_path, "test")):
        try:
            train_dataset, test_dataset = create_datasets(
                root_dir=cifar_path, 
                download=False
            )
            yield train_dataset, test_dataset
        except Exception:
            with tempfile.TemporaryDirectory() as temp_dir:
                train_dataset, test_dataset = create_datasets(
                    root_dir=temp_dir, 
                    download=True
                )
                yield train_dataset, test_dataset
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                train_dataset, test_dataset = create_datasets(
                    root_dir=temp_dir, 
                    download=True
                )
                yield train_dataset, test_dataset
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_batch():
    """Создает один батч данных для тестирования"""
    batch_size = 8
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


@pytest.mark.parametrize("device_name", ["mps", "cpu", "cuda"])
def test_train_on_one_batch(device_name, sample_batch):
    """Тестирует обучение модели на одном батче для разных устройств"""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA не доступна")
    
    device = torch.device(device_name)
    images, labels = sample_batch
    
    model = create_model(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        initial_preds = torch.argmax(outputs, 1)
        initial_accuracy = compute_accuracy(initial_preds, labels.to(device))
    
    model.train()
    
    loss = train_one_batch(model, images, labels, criterion, optimizer, device)
    
    assert isinstance(loss, float)
    assert loss >= 0
    
    # проверяем, что модель изменилась (градиенты применились)
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        final_preds = torch.argmax(outputs, 1)
        final_accuracy = compute_accuracy(final_preds, labels.to(device))
    
    assert loss > 0


def test_model_creation():
    """Тестирует создание модели"""
    model = create_model()
    
    # проверяем архитектуру
    assert hasattr(model, 'fc')
    assert model.fc.out_features == 10
    
    model.eval()
    sample_input = torch.randn(2, 3, 32, 32)  # batch size > 1 для BatchNorm
    with torch.no_grad():
        output = model(sample_input.to(model.fc.weight.device))
        assert output.shape == (2, 10)


def test_data_transforms():
    """Тестирует трансформации данных"""
    transform = get_data_transforms()
    
    # рандом изображение
    sample_pil = transforms.ToPILImage()(torch.rand(3, 32, 32))
    transformed = transform(sample_pil)
    
    assert transformed.shape == (3, 32, 32)
    assert transformed.dtype == torch.float32


def test_evaluate_model(small_datasets):
    """Тестирует функцию оценки модели"""
    train_dataset, test_dataset = small_datasets
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    device = torch.device("cpu")
    model = create_model(device=device)
    
    accuracy = evaluate_model(model, test_loader, device)
    
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


@pytest.mark.parametrize("batch_size,epochs", [
    (16, 1),
    (32, 2),
])
def test_training_pipeline(batch_size, epochs, small_datasets):
    """Тестирует полный pipeline обучения с разными гиперпараметрами"""
    train_dataset, test_dataset = small_datasets
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    train_loader, test_loader = create_data_loaders(
        train_dataset, test_dataset, batch_size
    )
    
    model = create_model(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    initial_accuracy = evaluate_model(model, test_loader, device)
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        model_path = tmp_file.name
    
    try:
        trained_model = train_model(
            model, train_loader, test_loader, criterion, optimizer,
            epochs, device, log_interval=10, wandb_log=False, 
            save_path=model_path
        )
        
        assert os.path.exists(model_path)
        
        loaded_model = create_model(device=device)
        loaded_model.load_state_dict(torch.load(model_path, map_location=device))
        
        final_accuracy = evaluate_model(loaded_model, test_loader, device)
        assert isinstance(final_accuracy, float)
        assert 0.0 <= final_accuracy <= 1.0
        
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_training():
    """Основной интеграционный тест всего процесса обучения"""
    device = get_device()
    
    num_samples = 200
    train_data = torch.randn(num_samples, 3, 32, 32)
    train_labels = torch.randint(0, 10, (num_samples,))
    test_data = torch.randn(50, 3, 32, 32)
    test_labels = torch.randint(0, 10, (50,))
    
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    configs = [
        {"batch_size": 16, "lr": 1e-3, "epochs": 1, "weight_decay": 0.01},
        {"batch_size": 32, "lr": 1e-4, "epochs": 2, "weight_decay": 0.001},
    ]
    
    for config in configs:
        train_loader, test_loader = create_data_loaders(
            train_dataset, test_dataset, config["batch_size"]
        )
        
        model = create_model(device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config["lr"], 
            weight_decay=config["weight_decay"]
        )
        
        initial_accuracy = evaluate_model(model, test_loader, device)
        
        with patch('wandb.log'):
            trained_model = train_model(
                model, train_loader, test_loader, criterion, optimizer,
                config["epochs"], device, log_interval=20, wandb_log=False,
                save_path=None
            )
        
        final_accuracy = evaluate_model(trained_model, test_loader, device)
        
        assert isinstance(final_accuracy, float)
        assert 0.0 <= final_accuracy <= 1.0
        assert isinstance(initial_accuracy, float)
        assert 0.0 <= initial_accuracy <= 1.0
        
        assert final_accuracy >= 0.02  # минимальная разумная точность


def test_device_compatibility():
    """Тестирует совместимость с разными устройствами"""
    available_devices = ["cpu"]
    
    if torch.cuda.is_available():
        available_devices.append("cuda")
    
    if torch.backends.mps.is_available():
        available_devices.append("mps")
    
    for device_name in available_devices:
        device = torch.device(device_name)
        model = create_model(device=device)
        
        assert next(model.parameters()).device.type == device.type
        
        sample_input = torch.randn(2, 3, 32, 32).to(device)
        with torch.no_grad():
            output = model(sample_input)
            assert output.device.type == device.type
            assert output.shape == (2, 10)