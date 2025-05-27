import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wandb
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm, trange
import os

from hparams import config


def compute_accuracy(preds, targets):
    result = (targets == preds).float().mean()
    return result


def get_device():
    """Получает доступное устройство для вычислений"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_model(num_classes=10, zero_init_residual=False, device=None):
    """Создаёт и инициализирует модель"""
    if device is None:
        device = get_device()
    
    model = resnet18(pretrained=False, num_classes=num_classes, zero_init_residual=zero_init_residual)
    model.to(device)
    return model


def get_data_transforms():
    """Возвращает трансформации для данных"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])


def create_datasets(root_dir='CIFAR10', download=False, transform=None):
    """Создаёт датасеты для обучения и тестирования"""
    if transform is None:
        transform = get_data_transforms()
    
    train_dataset = CIFAR10(
        root=os.path.join(root_dir, 'train'),
        train=True,
        transform=transform,
        download=download
    )
    
    test_dataset = CIFAR10(
        root=os.path.join(root_dir, 'test'),
        train=False,
        transform=transform,
        download=download
    )
    
    return train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, batch_size):
    """Создаёт data loaders"""
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size
    )
    
    return train_loader, test_loader


def train_one_batch(model, images, labels, criterion, optimizer, device):
    """Обучает модель на одном батче"""
    images = images.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate_model(model, test_loader, device):
    """Оценивает модель на тестовых данных"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.inference_mode():
        for test_images, test_labels in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            
            outputs = model(test_images)
            preds = torch.argmax(outputs, 1)
            
            all_preds.append(preds)
            all_labels.append(test_labels)
    
    accuracy = compute_accuracy(torch.cat(all_preds), torch.cat(all_labels))
    model.train()
    return accuracy.item()


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device, 
                log_interval=100, wandb_log=True, save_path="model.pt"):
    """Основная функция обучения модели"""
    for epoch in trange(epochs):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            loss = train_one_batch(model, images, labels, criterion, optimizer, device)
            
            if i % log_interval == 0:
                accuracy = evaluate_model(model, test_loader, device)
                
                if wandb_log:
                    metrics = {'test_acc': accuracy, 'train_loss': loss}
                    wandb.log(metrics, step=epoch * len(train_loader.dataset) + (i + 1) * train_loader.batch_size)
    
    if save_path:
        torch.save(model.state_dict(), save_path)
    
    return model


def main():
    wandb.init(config=config, project="effdl_example", name="baseline")
    
    device = get_device()
    
    train_dataset, test_dataset = create_datasets(download=False)
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, config["batch_size"])
    
    model = create_model(zero_init_residual=config["zero_init_residual"], device=device)
    wandb.watch(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )
    
    train_model(
        model, train_loader, test_loader, criterion, optimizer, 
        config["epochs"], device, log_interval=100, wandb_log=True
    )
    
    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)


if __name__ == '__main__':
    main()
