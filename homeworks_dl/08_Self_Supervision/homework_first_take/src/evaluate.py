"""
Модуль для оценки Prototypical Networks.
Включает функции тестирования и визуализации результатов.
"""

import os
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from .data import read_images, extract_sample, display_sample
from .model import load_protonet_conv


def test(model, test_x: np.ndarray, test_y: np.ndarray, 
         n_way: int, n_support: int, n_query: int, 
         test_episode: int) -> Tuple[float, float]:
    """
    # from omniglot_hw_from_ipynb.py: тестирует protonet
    Tests the protonet
    Args:
      model: trained model
      test_x (np.array): images of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
    Returns:
      tuple: (average loss, average accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        for episode in tqdm(range(test_episode), desc="Testing"):
            sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
            loss, output = model.set_forward_loss(sample)
            running_loss += output['loss']
            running_acc += output['acc']
        
    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    
    print(f'Test results -- Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}')
    return avg_loss, avg_acc


def visualize_predictions(sample: Dict, model, save_path: str = None) -> None:
    """
    Визуализирует предсказания модели на примере
    Args:
        sample: словарь с изображениями и параметрами эпизода
        model: обученная модель
        save_path: путь для сохранения изображения
    """
    model.eval()
    
    with torch.no_grad():
        # Получаем предсказания
        loss, output = model.set_forward_loss(sample)
        predictions = output['y_hat']
        
    # Извлекаем параметры
    images = sample['images']
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']
    
    # Создаем фигуру
    fig, axes = plt.subplots(n_way, n_support + n_query, 
                            figsize=(2 * (n_support + n_query), 2 * n_way))
    
    if n_way == 1:
        axes = axes.reshape(1, -1)
    
    pred_idx = 0
    for class_idx in range(n_way):
        for img_idx in range(n_support + n_query):
            ax = axes[class_idx, img_idx]
            
            # Получаем изображение
            img = images[class_idx, img_idx]
            
            # Денормализуем изображение для отображения
            img_display = img.permute(1, 2, 0).cpu().numpy()
            img_display = (img_display * np.array([0.229, 0.224, 0.225]) + 
                          np.array([0.485, 0.456, 0.406]))
            img_display = np.clip(img_display, 0, 1)
            
            ax.imshow(img_display)
            ax.axis('off')
            
            # Добавляем заголовок
            if img_idx < n_support:
                title = f"Support\nClass {class_idx}"
                ax.set_title(title, fontsize=8, color='blue')
            else:
                # Для query изображений показываем предсказание
                predicted_class = predictions[pred_idx]
                correct = predicted_class == class_idx
                color = 'green' if correct else 'red'
                title = f"Query\nTrue: {class_idx}\nPred: {predicted_class}"
                ax.set_title(title, fontsize=8, color=color)
                pred_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def generate_test_predictions(model, test_x: np.ndarray, test_y: np.ndarray,
                            n_way: int, n_support: int, n_query: int,
                            num_examples: int = 10) -> List[Dict]:
    """
    Генерирует предсказания для тестовых примеров
    Args:
        model: обученная модель
        test_x, test_y: тестовые данные
        n_way, n_support, n_query: параметры эпизода
        num_examples: количество примеров для генерации
    Returns:
        список словарей с результатами предсказаний
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in tqdm(range(num_examples), desc="Generating predictions"):
            sample = extract_sample(n_way, n_support, n_query, test_x, test_y, seed=i)
            loss, output = model.set_forward_loss(sample)
            
            # Создаем истинные метки для query set
            true_labels = []
            for class_idx in range(n_way):
                true_labels.extend([class_idx] * n_query)
            
            results.append({
                'episode': i,
                'loss': output['loss'],
                'accuracy': output['acc'],
                'true_labels': true_labels,
                'predictions': output['y_hat'].tolist(),
                'n_way': n_way,
                'n_support': n_support,
                'n_query': n_query
            })
    
    return results


def save_test_results(results: List[Dict], accuracy: float, 
                     save_dir: str = "metrics") -> None:
    """
    Сохраняет результаты тестирования в файлы
    Args:
        results: результаты предсказаний
        accuracy: общая точность
        save_dir: директория для сохранения
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Сохраняем общую точность
    with open(os.path.join(save_dir, "test_accuracy.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Number of test episodes: {len(results)}\n")
    
    # Сохраняем детальные предсказания
    with open(os.path.join(save_dir, "predictions_on_test.txt"), "w") as f:
        f.write("Episode\tTrue_Labels\tPredictions\tAccuracy\tLoss\n")
        for result in results:
            true_str = ",".join(map(str, result['true_labels']))
            pred_str = ",".join(map(str, result['predictions']))
            f.write(f"{result['episode']}\t{true_str}\t{pred_str}\t"
                   f"{result['accuracy']:.4f}\t{result['loss']:.4f}\n")
    
    print(f"Test results saved to {save_dir}/")


def load_model(model_path: str, device: str = None) -> torch.nn.Module:
    """
    Загружает обученную модель
    Args:
        model_path: путь к файлу модели
        device: устройство для загрузки
    Returns:
        загруженная модель
    """
    # Создаем модель
    model = load_protonet_conv(
        x_dim=(3, 28, 28),
        hid_dim=64,
        z_dim=64,
    )
    
    # Загружаем веса
    checkpoint = torch.load(model_path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {model_path}")
    print(f"Best training accuracy: {checkpoint.get('best_acc', 'N/A'):.4f}")
    
    return model


def main():
    """Основная функция для запуска тестирования"""
    parser = argparse.ArgumentParser(description='Evaluate Prototypical Networks')
    parser.add_argument('--model_path', type=str, default='models/protonet.pt',
                       help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--n_way', type=int, default=5,
                       help='Number of classes per episode')
    parser.add_argument('--n_support', type=int, default=5,
                       help='Number of support examples per class')
    parser.add_argument('--n_query', type=int, default=5,
                       help='Number of query examples per class')
    parser.add_argument('--test_episodes', type=int, default=1000,
                       help='Number of test episodes')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--save_dir', type=str, default='metrics',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("Loading test data...")
    # Загружаем тестовые данные
    test_x, test_y = read_images(f'{args.data_dir}/images_evaluation')
    print(f"Loaded test data: {test_x.shape}, {test_y.shape}")
    
    print("Loading model...")
    # Загружаем модель
    model = load_model(args.model_path)
    
    print("Running evaluation...")
    print(f"Parameters: n_way={args.n_way}, n_support={args.n_support}, n_query={args.n_query}")
    print(f"Test episodes: {args.test_episodes}")
    
    # Тестируем модель
    avg_loss, avg_acc = test(
        model=model,
        test_x=test_x,
        test_y=test_y,
        n_way=args.n_way,
        n_support=args.n_support,
        n_query=args.n_query,
        test_episode=args.test_episodes
    )
    
    # Генерируем предсказания для 10 примеров
    print("Generating predictions for 10 examples...")
    predictions = generate_test_predictions(
        model=model,
        test_x=test_x,
        test_y=test_y,
        n_way=args.n_way,
        n_support=args.n_support,
        n_query=args.n_query,
        num_examples=10
    )
    
    # Сохраняем результаты
    save_test_results(predictions, avg_acc, args.save_dir)
    
    # Визуализация если запрошена
    if args.visualize:
        print("Generating visualizations...")
        sample = extract_sample(args.n_way, args.n_support, args.n_query, 
                               test_x, test_y, seed=42)
        visualize_predictions(
            sample=sample,
            model=model,
            save_path=os.path.join(args.save_dir, "visualizations.png")
        )
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main() 