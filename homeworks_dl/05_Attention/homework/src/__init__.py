# HW03 Transformer Summarization package 
import torch
import logging
import sys
from datetime import datetime


def setup_logging(log_level=logging.INFO):
    """
    Настраивает логирование для всего проекта.
    
    Args:
        log_level: Уровень логирования (по умолчанию INFO)
    """
    # Создаем форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Удаляем существующие handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler для логов в файл
    file_handler = logging.FileHandler('training.log', mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Отключаем излишнее логирование от внешних библиотек
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def get_logger(name):
    """
    Получает логгер для конкретного модуля.
    
    Args:
        name: Имя модуля/логгера
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    return logging.getLogger(name)


# Автоматически настраиваем логирование при импорте пакета
setup_logging()


def get_device():
    """
    Выбирает оптимальное устройство по приоритету: mps -> cuda -> cpu.
    
    Returns:
        torch.device: Выбранное устройство
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu') 