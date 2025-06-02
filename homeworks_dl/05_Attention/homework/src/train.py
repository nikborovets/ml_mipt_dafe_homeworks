# from seminar_code: training loop with Label Smoothing (Task 5) and ROUGE metrics (Task 2)
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from rouge_score import rouge_scorer
import math
import os
import json
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from .data import load_processed_data_iterators, convert_batch
from .model import create_model, create_model_with_pretrained_embeddings
from . import get_device


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss (Задание 5).
    Сглаживает истинные метки для улучшения обобщения.
    """
    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        Применяет Label Smoothing к целевым меткам.
        
        Args:
            x: Логиты модели [batch_size, seq_len, vocab_size]
            target: Истинные метки [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Loss value
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    """Оптимизатор с warming-up."""
    def __init__(self, model_size, factor=2, warmup=4000, optimizer=None):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Обновляет параметры и расписание обучения."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Вычисляет текущий learning rate."""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * 
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))


def compute_rouge_scores(generated_texts, reference_texts):
    """
    Вычисляет ROUGE метрики (Задание 2).
    
    Args:
        generated_texts (list): Сгенерированные тексты
        reference_texts (list): Эталонные тексты
        
    Returns:
        dict: ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for gen, ref in zip(generated_texts, reference_texts):
        scores = scorer.score(ref, gen)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores)
    }


def do_epoch(model, criterion, data_iter, optimizer=None, name=None, field=None, writer=None, epoch=None):
    """
    Выполняет одну эпоху обучения или валидации.
    
    Args:
        model: Модель для обучения
        criterion: Функция потерь
        data_iter: Итератор данных
        optimizer: Оптимизатор (None для валидации)
        name: Название эпохи для логирования
        field: Word field для декодирования текстов
        writer: TensorBoard writer для логирования метрик
        epoch: Номер эпохи для TensorBoard
        
    Returns:
        tuple: (average_loss, rouge_scores)
    """
    is_train = optimizer is not None
    model.train(is_train)
    
    epoch_loss = 0
    total_tokens = 0
    
    # Для ROUGE метрик
    generated_texts = []
    reference_texts = []
    
    with tqdm(data_iter, desc=name) as pbar:
        for i, batch in enumerate(pbar):
            source_inputs, target_inputs, target_outputs, source_mask, target_mask = convert_batch(
                batch, pad_idx=field.vocab.stoi['<pad>']
            )
            
            # Прямой проход
            output = model(source_inputs, target_inputs, source_mask, target_mask)
            
            # Вычисление потерь
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                target_outputs.contiguous().view(-1)
            )
            
            # Проверка на NaN
            if torch.isnan(loss):
                print(f"\n❌ NaN detected in loss at batch {i}!")
                print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
                print(f"Target stats: min={target_outputs.min()}, max={target_outputs.max()}")
                if is_train:
                    print("Skipping this batch...")
                    continue
                else:
                    break
            
            if is_train:
                optimizer.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping для стабильности
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Проверка градиентов на NaN
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"\n❌ NaN gradient detected in {name}")
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print("Skipping optimizer step due to NaN gradients")
                    continue
                
                optimizer.step()
            
            # Подсчет статистик
            n_tokens = (target_outputs != field.vocab.stoi['<pad>']).data.sum().item()
            epoch_loss += loss.item()
            total_tokens += n_tokens
            
            # Логирование в TensorBoard каждые 100 батчей
            if writer is not None and epoch is not None and i % 100 == 0:
                step = epoch * len(data_iter) + i
                current_loss = loss.item() / n_tokens if n_tokens > 0 else 0
                mode = 'train' if is_train else 'val'
                writer.add_scalar(f'Loss/{mode}_batch', current_loss, step)
                writer.add_scalar(f'Learning_Rate', optimizer._rate if is_train and optimizer else 0, step)
                
                # Логируем статистики градиентов
                if is_train:
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    writer.add_scalar(f'Gradients/total_norm', total_norm, step)
            
            # Для ROUGE метрик (берем только первые несколько батчей, чтобы не замедлять)
            if i < 5 and field is not None:
                # Декодируем тексты для ROUGE
                pred_tokens = output.argmax(-1)
                for j in range(min(2, source_inputs.size(0))):  # Только первые 2 примера из батча
                    # Декодируем предсказание
                    pred_seq = pred_tokens[j].cpu().numpy()
                    pred_words = [field.vocab.itos[idx] for idx in pred_seq if idx not in [
                        field.vocab.stoi['<pad>'], field.vocab.stoi['<s>'], field.vocab.stoi['</s>']
                    ]]
                    
                    # Декодируем истинную последовательность
                    true_seq = target_outputs[j].cpu().numpy()
                    true_words = [field.vocab.itos[idx] for idx in true_seq if idx not in [
                        field.vocab.stoi['<pad>'], field.vocab.stoi['<s>'], field.vocab.stoi['</s>']
                    ]]
                    
                    generated_texts.append(' '.join(pred_words))
                    reference_texts.append(' '.join(true_words))
            
            current_loss = loss.item() / n_tokens if n_tokens > 0 else 0
            pbar.set_postfix(loss=current_loss, lr=optimizer._rate if is_train and optimizer else 0)
    
    # Вычисляем ROUGE метрики
    rouge_scores = {}
    if generated_texts and reference_texts:
        rouge_scores = compute_rouge_scores(generated_texts, reference_texts)
    
    avg_loss = epoch_loss / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, rouge_scores


def fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None, field=None, log_dir=None, 
        checkpoint_dir='checkpoints', save_every=1, resume_from_checkpoint=True, 
        save_best_model=True, early_stopping_patience=None, use_pretrained=False):
    """
    Обучает модель с поддержкой чекпоинтов.
    
    Args:
        model: Модель для обучения
        criterion: Функция потерь
        optimizer: Оптимизатор
        train_iter: Итератор тренировочных данных
        epochs_count: Количество эпох
        val_iter: Итератор валидационных данных
        field: Word field для декодирования
        log_dir: Директория для логов TensorBoard
        checkpoint_dir: Директория для чекпоинтов
        save_every: Сохранять чекпоинт каждые N эпох
        resume_from_checkpoint: Автоматически продолжать с последнего чекпоинта
        save_best_model: Сохранять лучшую модель по validation loss
        early_stopping_patience: Остановка при отсутствии улучшений (epochs)
        use_pretrained: Флаг использования предобученных эмбеддингов
        
    Returns:
        dict: История обучения
    """
    # Получаем устройство модели
    device = next(model.parameters()).device
    
    # Пытаемся загрузить последний чекпоинт
    start_epoch = 0
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_rouge': [],
        'val_rouge': [],
        'tensorboard_log_dir': None
    }
    
    if resume_from_checkpoint:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            start_epoch, history = load_checkpoint(latest_checkpoint, model, optimizer, device)
    
    # Создаем TensorBoard writer
    if log_dir is None:
        # Если продолжаем обучение, используем существующую директорию логов
        if history.get('tensorboard_log_dir') and resume_from_checkpoint:
            log_dir = history['tensorboard_log_dir']
            print(f"🔄 Resuming TensorBoard logging to existing directory: {log_dir}")
        else:
            # Создаем новую директорию с timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = "pretrained" if use_pretrained else "random"
            log_dir = f"runs/train_{model_type}_{timestamp}"
            print(f"📊 Creating new TensorBoard log directory: {log_dir}")
    
    writer = SummaryWriter(log_dir)
    history['tensorboard_log_dir'] = log_dir
    
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"Run 'tensorboard --logdir={log_dir}' to view logs")
    
    # Переменные для отслеживания лучшей модели
    best_val_loss = float('inf')
    best_model_path = None
    epochs_without_improvement = 0
    
    if history['val_losses']:
        best_val_loss = min(history['val_losses'])
    
    # Логируем архитектуру модели (только для нового обучения)
    if start_epoch == 0:
        try:
            # Создаем dummy input для визуализации графа модели
            vocab_size = len(field.vocab)
            dummy_src = torch.randint(1, vocab_size, (2, 10))
            dummy_tgt = torch.randint(1, vocab_size, (2, 8))
            dummy_src_mask = torch.ones(2, 1, 10).bool()
            dummy_tgt_mask = torch.ones(2, 1, 8).bool()
            
            device = next(model.parameters()).device
            dummy_src = dummy_src.to(device)
            dummy_tgt = dummy_tgt.to(device)
            dummy_src_mask = dummy_src_mask.to(device)
            dummy_tgt_mask = dummy_tgt_mask.to(device)
            
            writer.add_graph(model, (dummy_src, dummy_tgt, dummy_src_mask, dummy_tgt_mask))
        except Exception as e:
            print(f"Could not log model graph: {e}")
    
    for epoch in range(start_epoch, epochs_count):
        print(f"\nEpoch {epoch + 1}/{epochs_count}")
        
        # Обучение
        train_loss, train_rouge_scores = do_epoch(
            model, criterion, train_iter, optimizer, 
            name=f"Train Epoch {epoch + 1}", field=field,
            writer=writer, epoch=epoch
        )
        history['train_losses'].append(train_loss)
        history['train_rouge'].append(train_rouge_scores)
        
        print(f"Train Loss: {train_loss:.4f}")
        if train_rouge_scores:
            print(f"Train ROUGE-1: {train_rouge_scores.get('rouge1', 0):.4f}, "
                  f"ROUGE-2: {train_rouge_scores.get('rouge2', 0):.4f}, "
                  f"ROUGE-L: {train_rouge_scores.get('rougeL', 0):.4f}")
        
        # Валидация
        val_loss = None
        val_rouge_scores = None
        if val_iter is not None:
            val_loss, val_rouge_scores = do_epoch(
                model, criterion, val_iter, 
                name=f"Val Epoch {epoch + 1}", field=field,
                writer=writer, epoch=epoch
            )
            history['val_losses'].append(val_loss)
            history['val_rouge'].append(val_rouge_scores)
            
            print(f"Val Loss: {val_loss:.4f}")
            if val_rouge_scores:
                print(f"Val ROUGE-1: {val_rouge_scores.get('rouge1', 0):.4f}, "
                      f"ROUGE-2: {val_rouge_scores.get('rouge2', 0):.4f}, "
                      f"ROUGE-L: {val_rouge_scores.get('rougeL', 0):.4f}")
            
            # Проверяем улучшение модели
            if save_best_model and val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                # Сохраняем лучшую модель
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'val_rouge': val_rouge_scores
                }, best_model_path)
                print(f"💾 New best model saved! Val Loss: {val_loss:.4f}")
            else:
                epochs_without_improvement += 1
        
        # Логируем основные метрики в TensorBoard
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        if val_loss is not None:
            writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        
        if train_rouge_scores:
            for metric, value in train_rouge_scores.items():
                writer.add_scalar(f'ROUGE/train_{metric}', value, epoch)
        
        if val_rouge_scores:
            for metric, value in val_rouge_scores.items():
                writer.add_scalar(f'ROUGE/val_{metric}', value, epoch)
        
        # Логируем параметры модели (гистограммы весов)
        if epoch % 5 == 0:  # Каждые 5 эпох
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'Parameters/{name}', param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        # Сохранение чекпоинта
        if (epoch + 1) % save_every == 0 or epoch == epochs_count - 1:
            save_checkpoint(model, optimizer, epoch, history, checkpoint_dir)
            
            # Очистка старых чекпоинтов
            cleanup_old_checkpoints(checkpoint_dir, keep_last=3)
        
        # Early stopping
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            print(f"\n🛑 Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    writer.close()
    
    # Информация о результатах
    if best_model_path and os.path.exists(best_model_path):
        print(f"\n✅ Training completed!")
        print(f"Best model saved at: {best_model_path}")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    return history


def save_training_plot(history, save_path='training_plot.png'):
    """
    Сохраняет графики обучения.
    
    Args:
        history (dict): История обучения
        save_path (str): Путь для сохранения графика
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plots
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss')
    if history['val_losses']:
        ax1.plot(epochs, history['val_losses'], 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # ROUGE-1 plots
    if history['train_rouge'] and any(r for r in history['train_rouge']):
        train_rouge1 = [r.get('rouge1', 0) for r in history['train_rouge']]
        ax2.plot(epochs, train_rouge1, 'b-', label='Train ROUGE-1')
        if history['val_rouge']:
            val_rouge1 = [r.get('rouge1', 0) for r in history['val_rouge']]
            ax2.plot(epochs, val_rouge1, 'r-', label='Val ROUGE-1')
        ax2.set_title('ROUGE-1 Scores')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('ROUGE-1')
        ax2.legend()
        ax2.grid(True)
    
    # ROUGE-2 plots
    if history['train_rouge'] and any(r for r in history['train_rouge']):
        train_rouge2 = [r.get('rouge2', 0) for r in history['train_rouge']]
        ax3.plot(epochs, train_rouge2, 'b-', label='Train ROUGE-2')
        if history['val_rouge']:
            val_rouge2 = [r.get('rouge2', 0) for r in history['val_rouge']]
            ax3.plot(epochs, val_rouge2, 'r-', label='Val ROUGE-2')
        ax3.set_title('ROUGE-2 Scores')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('ROUGE-2')
        ax3.legend()
        ax3.grid(True)
    
    # ROUGE-L plots  
    if history['train_rouge'] and any(r for r in history['train_rouge']):
        train_rougeL = [r.get('rougeL', 0) for r in history['train_rouge']]
        ax4.plot(epochs, train_rougeL, 'b-', label='Train ROUGE-L')
        if history['val_rouge']:
            val_rougeL = [r.get('rougeL', 0) for r in history['val_rouge']]
            ax4.plot(epochs, val_rougeL, 'r-', label='Val ROUGE-L')
        ax4.set_title('ROUGE-L Scores')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('ROUGE-L')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to {save_path}")


def save_checkpoint(model, optimizer, epoch, history, checkpoint_dir='checkpoints', filename=None):
    """
    Сохраняет чекпоинт обучения.
    
    Args:
        model: Модель PyTorch
        optimizer: Оптимизатор (NoamOpt)
        epoch: Текущая эпоха
        history: История обучения
        checkpoint_dir: Директория для сохранения чекпоинтов
        filename: Имя файла (если None, генерируется автоматически)
        
    Returns:
        str: Путь к сохраненному файлу
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.optimizer.state_dict(),
        'optimizer_step': optimizer._step,
        'optimizer_rate': optimizer._rate,
        'history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Сохраняем также последний чекпоинт
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, device=None):
    """
    Загружает чекпоинт обучения.
    
    Args:
        checkpoint_path: Путь к файлу чекпоинта
        model: Модель PyTorch
        optimizer: Оптимизатор (NoamOpt)
        device: Устройство для загрузки (cuda/mps/cpu)
        
    Returns:
        tuple: (start_epoch, history)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, {
            'train_losses': [],
            'val_losses': [],
            'train_rouge': [],
            'val_rouge': [],
            'tensorboard_log_dir': None
        }
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Используем переданное устройство или CPU как fallback
    map_location = device if device is not None else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    
    # Загружаем состояние модели
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Загружаем состояние оптимизатора
    optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer._step = checkpoint['optimizer_step']
    optimizer._rate = checkpoint['optimizer_rate']
    
    start_epoch = checkpoint['epoch'] + 1  # Начинаем со следующей эпохи
    history = checkpoint['history']
    
    print(f"Checkpoint loaded from epoch {checkpoint['epoch']} on device: {map_location}")
    print(f"Resuming training from epoch {start_epoch}")
    
    return start_epoch, history


def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """
    Находит последний чекпоинт в директории.
    
    Args:
        checkpoint_dir: Директория с чекпоинтами
        
    Returns:
        str or None: Путь к последнему чекпоинту или None
    """
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_path):
        return latest_path
    
    # Ищем по номерам эпох
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    
    # Сортируем по номеру эпохи
    checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])


def cleanup_old_checkpoints(checkpoint_dir='checkpoints', keep_last=3):
    """
    Удаляет старые чекпоинты, оставляя только последние.
    
    Args:
        checkpoint_dir: Директория с чекпоинтами
        keep_last: Сколько последних чекпоинтов оставить
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    
    if len(checkpoint_files) <= keep_last:
        return
    
    # Сортируем по номеру эпохи
    checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    # Удаляем старые файлы
    for old_file in checkpoint_files[:-keep_last]:
        old_path = os.path.join(checkpoint_dir, old_file)
        os.remove(old_path)
        print(f"Removed old checkpoint: {old_path}")


def main():
    """Основная функция для запуска обучения."""
    # Парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Train Transformer model for text summarization')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--no-resume', action='store_true', help='Start training from scratch (ignore checkpoints)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory for checkpoints')
    parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--early-stopping', type=int, default=None, help='Early stopping patience (epochs)')
    parser.add_argument('--no-pretrained', action='store_true', help='Force use of random embeddings')
    
    args = parser.parse_args()
    
    print("Starting training...")
    print(f"Arguments: {vars(args)}")
    
    # Автоматический выбор устройства mps -> cuda -> cpu
    device = get_device()
    print(f"Using device: {device}")
    
    # Загружаем данные
    train_iter, test_iter, word_field = load_processed_data_iterators(device=device)
    vocab_size = len(word_field.vocab)
    
    # Проверяем наличие предобученных эмбеддингов
    fasttext_path = 'src/embeddings/cc.ru.300.bin'
    use_pretrained = os.path.exists(fasttext_path) and not args.no_pretrained
    
    if use_pretrained:
        print(f"✓ Found pretrained embeddings at {fasttext_path}")
        print("Training with pretrained Russian FastText embeddings (Task 6)")
        
        # Создание модели с предобученными эмбеддингами (300d)
        model = create_model_with_pretrained_embeddings(vocab_size, word_field, fasttext_path)
        model_size = 300
    else:
        if args.no_pretrained:
            print("⚠ Using random embeddings (forced by --no-pretrained flag)")
        else:
            print("⚠ Pretrained embeddings not found. Using random embeddings.")
            print("To use pretrained embeddings, run: python -m src.embeddings.download_embeddings")
        
        # Создание модели с обычными эмбеддингами (256d)
        model = create_model(vocab_size=vocab_size, d_model=256)
        model_size = 256
    
    model.to(device)
    
    print(f"Model created with d_model={model_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Label Smoothing Loss (Задание 5)
    criterion = LabelSmoothingLoss(
        size=vocab_size, 
        padding_idx=word_field.vocab.stoi['<pad>'], 
        smoothing=0.1
    )
    
    # Оптимизатор с учетом размерности модели
    if use_pretrained:
        # Более консервативный learning rate для предобученных эмбеддингов
        optimizer = NoamOpt(
            model_size=model_size, factor=1, warmup=4000,  # Уменьшили factor с 2 до 1
            optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        )
        print("Using conservative learning rate for pretrained embeddings (factor=1)")
    else:
        optimizer = NoamOpt(
            model_size=model_size, factor=2, warmup=4000,
            optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        )
        print("Using standard learning rate for random embeddings (factor=2)")
    
    # Создаем необходимые директории
    os.makedirs('runs', exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Проверяем существующие чекпоинты
    if not args.no_resume:
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint:
            print(f"🔄 Found existing checkpoint: {latest_checkpoint}")
            print("Training will resume from the last checkpoint.")
            print("Use --no-resume to start from scratch.")
        else:
            print("🆕 No existing checkpoints found. Starting fresh training.")
    else:
        print("🆕 Starting training from scratch (--no-resume flag)")
    
    # Обучение с чекпоинтами
    history = fit(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_iter=train_iter,
        epochs_count=args.epochs,
        val_iter=test_iter,
        field=word_field,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        resume_from_checkpoint=not args.no_resume,
        save_best_model=True,
        early_stopping_patience=args.early_stopping,
        use_pretrained=use_pretrained
    )
    
    # Сохранение финальных результатов
    model_suffix = "_pretrained" if use_pretrained else "_random"
    
    # Загружаем лучшую модель для финального сохранения
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print(f"📥 Loading best model from {best_model_path}")
        best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Сохраняем в стандартном формате
        torch.save(model.state_dict(), f'best_model{model_suffix}.pt')
        print(f"💾 Best model saved as: best_model{model_suffix}.pt")
        
        # Информация о лучшей модели
        best_epoch = best_checkpoint['epoch']
        best_val_loss = best_checkpoint['val_loss']
        best_rouge = best_checkpoint.get('val_rouge', {})
        
        print(f"📊 Best model from epoch {best_epoch + 1}:")
        print(f"   Validation Loss: {best_val_loss:.4f}")
        if best_rouge:
            for metric, score in best_rouge.items():
                print(f"   {metric.upper()}: {score:.4f}")
    else:
        # Сохраняем текущую модель если лучшей нет
        torch.save(model.state_dict(), f'best_model{model_suffix}.pt')
        print(f"💾 Final model saved as: best_model{model_suffix}.pt")
    
    # Сохраняем графики обучения
    save_training_plot(history, f'training_plot{model_suffix}.png')
    
    # Сохраняем информацию о модели
    model_info = {
        'use_pretrained': use_pretrained,
        'model_size': model_size,
        'vocab_size': vocab_size,
        'fasttext_path': fasttext_path if use_pretrained else None,
        'total_epochs': len(history['train_losses']),
        'final_train_loss': history['train_losses'][-1] if history['train_losses'] else None,
        'final_val_loss': history['val_losses'][-1] if history['val_losses'] else None,
        'final_rouge': history['val_rouge'][-1] if history['val_rouge'] else None,
        'best_val_loss': best_val_loss if 'best_val_loss' in locals() else None,
        'tensorboard_log_dir': history.get('tensorboard_log_dir'),
        'checkpoint_dir': args.checkpoint_dir,
        'training_args': vars(args)
    }
    
    with open(f'model_info{model_suffix}.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED!")
    print("="*60)
    print(f"Model type: {'Pretrained embeddings' if use_pretrained else 'Random embeddings'}")
    print(f"Total epochs trained: {len(history['train_losses'])}")
    print(f"Model saved as: best_model{model_suffix}.pt")
    print(f"Training plot saved as: training_plot{model_suffix}.png")
    print(f"Model info saved as: model_info{model_suffix}.json")
    print(f"Checkpoints saved in: {args.checkpoint_dir}/")
    print(f"TensorBoard logs: {history.get('tensorboard_log_dir')}")
    print(f"Run 'tensorboard --logdir={history.get('tensorboard_log_dir')}' to view training progress")
    
    # Задание 6: Сравнение результатов
    if use_pretrained and model_info['final_rouge']:
        print("\n" + "="*60)
        print("TASK 6: Results with pretrained Russian embeddings")
        print("="*60)
        print(f"Final ROUGE scores:")
        for metric, score in model_info['final_rouge'].items():
            print(f"  {metric.upper()}: {score:.4f}")
        print(f"Final validation loss: {model_info['final_val_loss']:.4f}")
        if 'best_val_loss' in locals():
            print(f"Best validation loss: {best_val_loss:.4f}")
        print("\nTo compare with random embeddings, use --no-pretrained flag.")
    
    print("\n💡 Tips:")
    print("  - Resume training: python -m src.train")
    print("  - Start fresh: python -m src.train --no-resume")
    print("  - More epochs: python -m src.train --epochs 20")
    print("  - Early stopping: python -m src.train --early-stopping 5")


if __name__ == "__main__":
    main() 