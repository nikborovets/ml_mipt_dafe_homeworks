import pytest

from hparams import config


def test_config_structure():
    """Тестирует структуру конфигурации"""
    assert isinstance(config, dict)
    
    required_keys = [
        "batch_size", "learning_rate", "weight_decay", 
        "epochs", "zero_init_residual"
    ]
    
    for key in required_keys:
        assert key in config, f"Отсутствует ключ {key} в конфигурации"


def test_config_values():
    """Тестирует корректность значений в конфигурации"""
    assert isinstance(config["batch_size"], int)
    assert config["batch_size"] > 0
    
    assert isinstance(config["learning_rate"], (int, float))
    assert config["learning_rate"] > 0
    
    assert isinstance(config["weight_decay"], (int, float))
    assert config["weight_decay"] >= 0
    
    assert isinstance(config["epochs"], int)
    assert config["epochs"] > 0
    
    assert isinstance(config["zero_init_residual"], bool)


def test_config_default_values():
    """Тестирует конкретные значения по умолчанию"""
    assert config["batch_size"] == 64
    assert config["learning_rate"] == 1e-5
    assert config["weight_decay"] == 0.01
    assert config["epochs"] == 2
    assert config["zero_init_residual"] == False


def test_config_ranges():
    """Тестирует разумные диапазоны значений"""
    assert 1 <= config["batch_size"] <= 1024
    assert 1e-6 <= config["learning_rate"] <= 1e-1
    assert 0 <= config["weight_decay"] <= 1
    assert 1 <= config["epochs"] <= 1000


@pytest.mark.parametrize("key,expected_type", [
    ("batch_size", int),
    ("learning_rate", (int, float)),
    ("weight_decay", (int, float)), 
    ("epochs", int),
    ("zero_init_residual", bool),
])
def test_config_types_parametrized(key, expected_type):
    assert isinstance(config[key], expected_type) 