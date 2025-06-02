# from seminar_code: training loop with Label Smoothing (Task 5) and ROUGE metrics (Task 2)
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from rouge_score import rouge_scorer

from .data import convert_batch
from .model import create_model


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


def do_epoch(model, criterion, data_iter, optimizer=None, name=None, field=None):
    """
    Выполняет одну эпоху обучения или валидации.
    
    Args:
        model: Модель для обучения
        criterion: Функция потерь
        data_iter: Итератор данных
        optimizer: Оптимизатор (None для валидации)
        name: Название эпохи для логирования
        field: Word field для декодирования текстов
        
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
            
            if is_train:
                optimizer.optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Подсчет статистик
            n_tokens = (target_outputs != field.vocab.stoi['<pad>']).data.sum().item()
            epoch_loss += loss.item()
            total_tokens += n_tokens
            
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
            
            pbar.set_postfix(loss=loss.item() / n_tokens if n_tokens > 0 else 0)
    
    # Вычисляем ROUGE метрики
    rouge_scores = {}
    if generated_texts and reference_texts:
        rouge_scores = compute_rouge_scores(generated_texts, reference_texts)
    
    avg_loss = epoch_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, rouge_scores


def fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None, field=None):
    """
    Обучает модель.
    
    Args:
        model: Модель для обучения
        criterion: Функция потерь
        optimizer: Оптимизатор
        train_iter: Итератор тренировочных данных
        epochs_count: Количество эпох
        val_iter: Итератор валидационных данных
        field: Word field для декодирования
        
    Returns:
        dict: История обучения
    """
    train_losses = []
    val_losses = []
    train_rouge = []
    val_rouge = []
    
    for epoch in range(epochs_count):
        print(f"\nEpoch {epoch + 1}/{epochs_count}")
        
        # Обучение
        train_loss, train_rouge_scores = do_epoch(
            model, criterion, train_iter, optimizer, 
            name=f"Train Epoch {epoch + 1}", field=field
        )
        train_losses.append(train_loss)
        train_rouge.append(train_rouge_scores)
        
        print(f"Train Loss: {train_loss:.4f}")
        if train_rouge_scores:
            print(f"Train ROUGE-1: {train_rouge_scores.get('rouge1', 0):.4f}, "
                  f"ROUGE-2: {train_rouge_scores.get('rouge2', 0):.4f}, "
                  f"ROUGE-L: {train_rouge_scores.get('rougeL', 0):.4f}")
        
        # Валидация
        if val_iter is not None:
            val_loss, val_rouge_scores = do_epoch(
                model, criterion, val_iter, 
                name=f"Val Epoch {epoch + 1}", field=field
            )
            val_losses.append(val_loss)
            val_rouge.append(val_rouge_scores)
            
            print(f"Val Loss: {val_loss:.4f}")
            if val_rouge_scores:
                print(f"Val ROUGE-1: {val_rouge_scores.get('rouge1', 0):.4f}, "
                      f"ROUGE-2: {val_rouge_scores.get('rouge2', 0):.4f}, "
                      f"ROUGE-L: {val_rouge_scores.get('rougeL', 0):.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_rouge': train_rouge,
        'val_rouge': val_rouge
    }


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


def main():
    """Основная функция для запуска обучения."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Загрузка данных
    from .data import load_and_process_data
    train_iter, test_iter, word_field = load_and_process_data(device=device)
    
    # Создание модели
    vocab_size = len(word_field.vocab)
    model = create_model(vocab_size=vocab_size)
    model.to(device)
    
    # Label Smoothing Loss (Задание 5)
    criterion = LabelSmoothingLoss(
        size=vocab_size, 
        padding_idx=word_field.vocab.stoi['<pad>'], 
        smoothing=0.1
    )
    
    # Оптимизатор
    optimizer = NoamOpt(
        model_size=256, factor=2, warmup=4000,
        optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )
    
    # Обучение
    print("Starting training...")
    history = fit(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_iter=train_iter,
        epochs_count=10,
        val_iter=test_iter,
        field=word_field
    )
    
    # Сохранение результатов
    torch.save(model.state_dict(), 'best_model.pt')
    save_training_plot(history, 'training_plot.png')
    
    print("Training completed!")


if __name__ == "__main__":
    main() 