# HW03 Transformer Summarization package 
import torch


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