"""
Модуль с моделями для Prototypical Networks.
Включает CNN энкодер и класс ProtoNet.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    # from omniglot_hw_from_ipynb.py: CNN энкодер для изображений Omniglot
    CNN encoder состоящий из 4 блоков Conv+BN+ReLU+MaxPool
    """
    
    def __init__(self, x_dim: Tuple[int, int, int] = (3, 28, 28), 
                 hid_dim: int = 64, z_dim: int = 64):
        """
        Args:
            x_dim: размерность входного изображения (channels, height, width)
            hid_dim: количество фильтров в сверточных слоях
            z_dim: размерность выходного вектора
        """
        super(CNNEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # Блок 1: Conv + BN + ReLU + MaxPool
            nn.Conv2d(x_dim[0], hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Блок 2: Conv + BN + ReLU + MaxPool  
            nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Блок 3: Conv + BN + ReLU + MaxPool
            nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Блок 4: Conv + BN + ReLU + MaxPool
            nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Вычисляем размер после сверток для полносвязного слоя
        # Для входа 28x28 после 4 MaxPool(2) получаем 1x1
        self.fc = nn.Linear(hid_dim * 1 * 1, z_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass энкодера
        Args:
            x: входные изображения [batch_size, channels, height, width]
        Returns:
            encoded features [batch_size, z_dim]
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x


class ProtoNet(nn.Module):
    """
    # from omniglot_hw_from_ipynb.py: Prototypical Network для few-shot learning
    """
    
    def __init__(self, encoder: nn.Module):
        """
        Args:
            encoder: CNN encoding the images in sample
        """
        super(ProtoNet, self).__init__()
        # Определяем устройство (mps для M1 Mac, cuda для GPU, cpu для остального)
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.encoder = encoder.to(self.device)

    def set_forward_loss(self, sample: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        # from omniglot_hw_from_ipynb.py: вычисляет loss, accuracy и output для задачи классификации
        Computes loss, accuracy and output for classification task
        Args:
            sample (dict): содержит 'images', 'n_way', 'n_support', 'n_query'
        Returns:
            torch.Tensor: loss value
            dict: содержит loss, accuracy и predictions
        """
        sample_images = sample['images'].to(self.device)
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']
        
        # Размерности: [n_way, n_support + n_query, channels, height, width]
        # Преобразуем в [n_way * (n_support + n_query), channels, height, width]
        x = sample_images.view(-1, *sample_images.shape[2:])
        
        # Получаем embeddings для всех изображений
        z = self.encoder(x)  # [n_way * (n_support + n_query), z_dim]
        
        # Разделяем на support и query
        z_dim = z.size(-1)
        z = z.view(n_way, n_support + n_query, z_dim)
        
        z_support = z[:, :n_support]  # [n_way, n_support, z_dim]
        z_query = z[:, n_support:]    # [n_way, n_query, z_dim]
        
        # Вычисляем прототипы классов (среднее по support set)
        z_proto = z_support.mean(dim=1)  # [n_way, z_dim]
        
        # Вычисляем расстояния от query до прототипов
        z_query_flat = z_query.reshape(n_way * n_query, z_dim)  # [n_way * n_query, z_dim]
        
        # Евклидово расстояние
        dists = torch.cdist(z_query_flat, z_proto)  # [n_way * n_query, n_way]
        
        # Применяем softmax к отрицательным расстояниям
        log_p_y = F.log_softmax(-dists, dim=1)  # [n_way * n_query, n_way]
        
        # Создаем метки для query set
        target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        target_inds = target_inds.reshape(-1).to(self.device)  # [n_way * n_query]
        
        # Вычисляем loss
        loss_val = F.nll_loss(log_p_y, target_inds)
        
        # Вычисляем accuracy
        _, y_hat = log_p_y.max(1)
        acc_val = torch.eq(y_hat, target_inds).float().mean()
        
        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat.cpu().numpy()
        }


def load_protonet_conv(x_dim: Tuple[int, int, int] = (3, 28, 28), 
                      hid_dim: int = 64, z_dim: int = 64) -> ProtoNet:
    """
    # from omniglot_hw_from_ipynb.py: загружает prototypical network model
    Loads the prototypical network model
    Args:
      x_dim (tuple): dimension of input image
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded image
    Returns:
      Model (ProtoNet): экземпляр ProtoNet
    """
    encoder = CNNEncoder(x_dim, hid_dim, z_dim)
    return ProtoNet(encoder)


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет евклидово расстояние между двумя тензорами
    Args:
        x: первый тензор [N, D]
        y: второй тензор [M, D]
    Returns:
        расстояния [N, M]
    """
    return torch.cdist(x, y)


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет косинусное расстояние между двумя тензорами
    Args:
        x: первый тензор [N, D]
        y: второй тензор [M, D]
    Returns:
        расстояния [N, M]
    """
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return 1 - torch.mm(x_norm, y_norm.t()) 