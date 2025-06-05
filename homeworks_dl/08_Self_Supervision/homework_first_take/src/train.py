"""
Модуль для обучения Prototypical Networks.
Включает функции обучения с TensorBoard логированием.
"""

import os
import argparse
from typing import Optional
import yaml
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data import read_images, extract_sample
from .model import load_protonet_conv


def train(model, optimizer, train_x: np.ndarray, train_y: np.ndarray, 
          n_way: int, n_support: int, n_query: int, 
          max_epoch: int, epoch_size: int,
          writer: Optional[SummaryWriter] = None,
          save_path: str = "models/protonet.pt") -> None:
    """
    # from omniglot_hw_from_ipynb.py: обучает protonet с добавлением TensorBoard логирования
    Trains the protonet
    Args:
      model: модель ProtoNet
      optimizer: оптимизатор
      train_x (np.array): images of training set
      train_y (np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      max_epoch (int): max epochs to train on
      epoch_size (int): episodes per epoch
      writer: TensorBoard writer для логирования
      save_path: путь для сохранения лучшей модели
    """
    # divide the learning rate by 2 at each epoch, as suggested in paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop
    best_acc = 0.0  # лучшая accuracy для сохранения модели
    
    # Создаем директорию для сохранения модели
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        # Используем tqdm для прогресс-бара
        pbar = tqdm(range(epoch_size), desc=f"Epoch {epoch + 1}/{max_epoch}")
        
        for episode in pbar:
            sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            running_loss += output['loss']
            running_acc += output['acc']
            loss.backward()
            optimizer.step()
            
            # Обновляем прогресс-бар
            pbar.set_postfix({
                'Loss': f"{output['loss']:.4f}",
                'Acc': f"{output['acc']:.4f}"
            })
        
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        
        print(f'Epoch {epoch+1:d} -- Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # TensorBoard логирование
        if writer is not None:
            writer.add_scalar('Train/Loss', epoch_loss, epoch)
            writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
            writer.add_scalar('Train/Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        # Сохраняем лучшую модель
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'loss': epoch_loss,
            }, save_path)
            print(f"Saved best model with accuracy: {best_acc:.4f}")
        
        epoch += 1
        scheduler.step()
    
    print(f"Training completed. Best accuracy: {best_acc:.4f}")


def validate(model, val_x: np.ndarray, val_y: np.ndarray,
            n_way: int, n_support: int, n_query: int,
            val_episodes: int = 100) -> tuple:
    """
    Валидация модели
    Args:
        model: обученная модель
        val_x, val_y: валидационные данные
        n_way, n_support, n_query: параметры эпизода
        val_episodes: количество эпизодов для валидации
    Returns:
        tuple: (средний loss, средняя accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        for episode in tqdm(range(val_episodes), desc="Validation"):
            sample = extract_sample(n_way, n_support, n_query, val_x, val_y)
            loss, output = model.set_forward_loss(sample)
            running_loss += output['loss']
            running_acc += output['acc']
    
    model.train()
    avg_loss = running_loss / val_episodes
    avg_acc = running_acc / val_episodes
    
    return avg_loss, avg_acc


def main():
    """Основная функция для запуска обучения"""
    parser = argparse.ArgumentParser(description='Train Prototypical Networks')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--n_way', type=int, default=60,
                       help='Number of classes per episode')
    parser.add_argument('--n_support', type=int, default=5,
                       help='Number of support examples per class')
    parser.add_argument('--n_query', type=int, default=5,
                       help='Number of query examples per class')
    parser.add_argument('--max_epoch', type=int, default=5,
                       help='Maximum number of epochs')
    parser.add_argument('--epoch_size', type=int, default=2000,
                       help='Number of episodes per epoch')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--log_dir', type=str, default='runs/omniglot_protonet',
                       help='TensorBoard log directory')
    parser.add_argument('--save_path', type=str, default='models/protonet.pt',
                       help='Path to save the best model')
    
    args = parser.parse_args()
    
    # Загружаем конфиг если есть (только если аргумент не был передан в CLI)
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Получаем аргументы по умолчанию
        default_parser = argparse.ArgumentParser()
        default_parser.add_argument('--n_way', type=int, default=60)
        default_parser.add_argument('--n_support', type=int, default=5)
        default_parser.add_argument('--n_query', type=int, default=5)
        default_parser.add_argument('--max_epoch', type=int, default=5)
        default_parser.add_argument('--epoch_size', type=int, default=2000)
        default_parser.add_argument('--lr', type=float, default=0.001)
        default_parser.add_argument('--log_dir', type=str, default='runs/omniglot_protonet')
        default_parser.add_argument('--save_path', type=str, default='models/protonet.pt')
        default_args = default_parser.parse_args([])
        
        # Обновляем только те аргументы, которые равны значениям по умолчанию
        for key, value in config.items():
            if hasattr(args, key) and hasattr(default_args, key):
                if getattr(args, key) == getattr(default_args, key):
                    setattr(args, key, value)
    
    print("Loading data...")
    # Загружаем данные
    train_x, train_y = read_images(f'{args.data_dir}/images_background')
    print(f"Loaded training data: {train_x.shape}, {train_y.shape}")
    
    # Создаем модель
    print("Creating model...")
    model = load_protonet_conv(
        x_dim=(3, 28, 28),
        hid_dim=64,
        z_dim=64,
    )
    
    # Создаем оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Создаем новую директорию с timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/train_{timestamp}"
    print(f"📊 Creating new TensorBoard log directory: {log_dir}")
    
    # Создаем TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    print("Starting training...")
    print(f"Device: {model.device}")
    print(f"Parameters: n_way={args.n_way}, n_support={args.n_support}, n_query={args.n_query}")
    print(f"Training: max_epoch={args.max_epoch}, epoch_size={args.epoch_size}")
    
    # Обучаем модель
    train(
        model=model,
        optimizer=optimizer,
        train_x=train_x,
        train_y=train_y,
        n_way=args.n_way,
        n_support=args.n_support,
        n_query=args.n_query,
        max_epoch=args.max_epoch,
        epoch_size=args.epoch_size,
        writer=writer,
        save_path=args.save_path
    )
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main() 