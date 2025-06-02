# from seminar_code: generation and evaluation with attention visualization (Tasks 1, 3)
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm
import os
from rouge_score import rouge_scorer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from .data import load_and_process_data, make_mask, subsequent_mask
from .model import create_model
from .train import compute_rouge_scores
from . import get_device


def generate_summary(model, field, src_text, max_len=50, device=None):
    """
    Генерирует суммаризацию для входного текста (Задание 1).
    
    Args:
        model: Обученная модель
        field: Word field для токенизации
        src_text (str): Исходный текст
        max_len (int): Максимальная длина суммаризации
        device: Устройство для вычислений. Если None, выбирается автоматически
        
    Returns:
        tuple: (generated_text, attention_weights)
    """
    if device is None:
        device = get_device()
    
    model.eval()
    model = model.to(device)
    
    # Токенизация входного текста
    src_tokens = field.preprocess(src_text)
    src_indices = [field.vocab.stoi[token] for token in src_tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    # Создание маски для источника
    src_mask = (src_tensor != field.vocab.stoi['<pad>']).unsqueeze(-2)
    
    # Энкодинг
    encoder_output = model.encode(src_tensor, src_mask)
    
    # Инициализация декодера
    ys = torch.ones(1, 1).fill_(field.vocab.stoi['<s>']).type_as(src_tensor.data)
    
    attention_weights = []
    
    for i in range(max_len):
        # Создание маски для цели
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src_tensor.data)
        
        # Декодинг
        decoder_output = model.decode(ys, encoder_output, src_mask, tgt_mask)
        
        # Получение следующего токена
        prob = model.generator(decoder_output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        
        # Добавление токена к последовательности
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word)], dim=1)
        
        # Остановка при достижении EOS токена
        if next_word == field.vocab.stoi['</s>']:
            break
    
    # Декодирование результата
    generated_tokens = ys[0].cpu().numpy()
    generated_words = [field.vocab.itos[idx] for idx in generated_tokens[1:] if idx not in [
        field.vocab.stoi['<pad>'], field.vocab.stoi['<s>'], field.vocab.stoi['</s>']
    ]]
    
    return ' '.join(generated_words), attention_weights


def extract_attention_weights(model, field, src_text, tgt_text, device=None):
    """
    Извлекает веса внимания для визуализации (Задание 3).
    
    Args:
        model: Обученная модель
        field: Word field
        src_text (str): Исходный текст
        tgt_text (str): Целевой текст
        device: Устройство для вычислений. Если None, выбирается автоматически
        
    Returns:
        tuple: (attention_weights, src_tokens, tgt_tokens)
    """
    if device is None:
        device = get_device()
    
    model.eval()
    model = model.to(device)
    
    # Токенизация
    src_tokens = field.preprocess(src_text)
    tgt_tokens = field.preprocess(tgt_text)
    
    src_indices = [field.vocab.stoi[token] for token in src_tokens]
    tgt_indices = [field.vocab.stoi[token] for token in tgt_tokens]
    
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
    
    # Создание масок
    src_mask = (src_tensor != field.vocab.stoi['<pad>']).unsqueeze(-2)
    tgt_mask = (tgt_tensor != field.vocab.stoi['<pad>']).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt_tensor.size(-1)).type_as(tgt_mask.data)
    
    # Прямой проход с сохранением attention weights
    with torch.no_grad():
        encoder_output = model.encode(src_tensor, src_mask)
        
        # Получаем attention weights из последнего слоя декодера
        x = model.decoder.pe(model.decoder.embeddings(tgt_tensor))
        
        attention_weights = []
        for layer in model.decoder.layers:
            # Self-attention
            x = layer.sublayer[0](x, lambda x: layer.self_attn(x, x, x, tgt_mask)[0])
            # Encoder-decoder attention
            x, attn = layer.encoder_attn(x, encoder_output, encoder_output, src_mask)
            attention_weights.append(attn.cpu().numpy())
            x = layer.sublayer[1](x, lambda x: layer.encoder_attn(x, encoder_output, encoder_output, src_mask)[0])
            x = layer.sublayer[2](x, layer.feed_forward)
    
    return attention_weights, src_tokens, tgt_tokens


def plot_attention(attention_weights, src_tokens, tgt_tokens, layer_idx=0, head_idx=0, save_path=None):
    """
    Визуализирует матрицы внимания (Задание 3).
    
    Args:
        attention_weights: Веса внимания
        src_tokens: Токены источника
        tgt_tokens: Токены цели
        layer_idx: Индекс слоя
        head_idx: Индекс головы внимания
        save_path: Путь для сохранения
    """
    if not attention_weights:
        print("No attention weights available")
        return
    
    # Выбираем конкретный слой и голову
    attn = attention_weights[layer_idx][0, head_idx]  # [tgt_len, src_len]
    
    # Обрезаем до реальной длины последовательностей
    attn = attn[:len(tgt_tokens), :len(src_tokens)]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(attn, 
                xticklabels=src_tokens, 
                yticklabels=tgt_tokens,
                cmap='Blues',
                cbar=True)
    
    plt.title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_model_on_test(model, field, test_iter, device=None, num_examples=10):
    """
    Оценивает модель на тестовых данных с ROUGE метриками.
    
    Args:
        model: Обученная модель
        field: Word field
        test_iter: Итератор тестовых данных
        device: Устройство для вычислений. Если None, выбирается автоматически
        num_examples: Количество примеров для оценки
        
    Returns:
        dict: Результаты оценки
    """
    if device is None:
        device = get_device()
    
    model.eval()
    
    generated_texts = []
    reference_texts = []
    examples_data = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            if i >= num_examples:
                break
                
            # Берем только первый пример из батча
            source = batch.source[:, 0].cpu().numpy()
            target = batch.target[:, 0].cpu().numpy()
            
            # Декодируем исходный текст
            src_words = [field.vocab.itos[idx] for idx in source if idx not in [
                field.vocab.stoi['<pad>'], field.vocab.stoi['<s>'], field.vocab.stoi['</s>']
            ]]
            src_text = ' '.join(src_words)
            
            # Декодируем целевой текст
            tgt_words = [field.vocab.itos[idx] for idx in target if idx not in [
                field.vocab.stoi['<pad>'], field.vocab.stoi['<s>'], field.vocab.stoi['</s>']
            ]]
            tgt_text = ' '.join(tgt_words)
            
            # Генерируем суммаризацию
            generated_text, _ = generate_summary(model, field, src_text, device=device)
            
            generated_texts.append(generated_text)
            reference_texts.append(tgt_text)
            
            examples_data.append({
                'source': src_text,
                'reference': tgt_text,
                'generated': generated_text
            })
    
    # Вычисляем ROUGE метрики
    rouge_scores = compute_rouge_scores(generated_texts, reference_texts)
    
    return {
        'rouge_scores': rouge_scores,
        'examples': examples_data
    }


def create_attention_examples(model, field, test_iter, device=None, num_examples=3):
    """
    Создает примеры визуализации внимания (Задание 3).
    
    Args:
        model: Обученная модель
        field: Word field
        test_iter: Итератор тестовых данных
        device: Устройство для вычислений. Если None, выбирается автоматически
        num_examples: Количество примеров
    """
    if device is None:
        device = get_device()
    
    os.makedirs('docs/attention_examples', exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            if i >= num_examples:
                break
                
            # Берем первый пример из батча
            source = batch.source[:, 0].cpu().numpy()
            target = batch.target[:, 0].cpu().numpy()
            
            # Декодируем тексты
            src_words = [field.vocab.itos[idx] for idx in source if idx not in [
                field.vocab.stoi['<pad>'], field.vocab.stoi['<s>'], field.vocab.stoi['</s>']
            ]]
            tgt_words = [field.vocab.itos[idx] for idx in target if idx not in [
                field.vocab.stoi['<pad>'], field.vocab.stoi['<s>'], field.vocab.stoi['</s>']
            ]]
            
            src_text = ' '.join(src_words)
            tgt_text = ' '.join(tgt_words)
            
            # Извлекаем веса внимания
            attention_weights, src_tokens, tgt_tokens = extract_attention_weights(
                model, field, src_text, tgt_text, device
            )
            
            # Создаем визуализации для разных слоев и голов
            for layer_idx in range(min(2, len(attention_weights))):
                for head_idx in range(min(2, attention_weights[layer_idx].shape[1])):
                    save_path = f'docs/attention_examples/example_{i+1}_layer_{layer_idx}_head_{head_idx}.png'
                    plot_attention(attention_weights, src_tokens, tgt_tokens, 
                                 layer_idx, head_idx, save_path)
            
            print(f"Example {i+1}:")
            print(f"Source: {src_text[:100]}...")
            print(f"Target: {tgt_text}")
            print()


def main():
    """Основная функция для оценки модели."""
    print("Starting evaluation...")
    
    # Автоматический выбор устройства mps -> cuda -> cpu
    device = get_device()
    print(f"Using device: {device}")
    
    # Загружаем данные и модель
    train_iter, test_iter, word_field = load_and_process_data(device=device)
    
    # Создание и загрузка модели
    vocab_size = len(word_field.vocab)
    model = create_model(vocab_size=vocab_size)
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.to(device)
    model.eval()
    
    print("Evaluating model on test data...")
    
    # Оценка на тестовых данных
    results = evaluate_model_on_test(model, word_field, test_iter, device, num_examples=10)
    
    print("ROUGE Scores:")
    print(f"ROUGE-1: {results['rouge_scores']['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge_scores']['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rouge_scores']['rougeL']:.4f}")
    
    # Создаем TensorBoard writer для логирования результатов оценки
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_log_dir = f"runs/evaluation_{timestamp}"
    writer = SummaryWriter(eval_log_dir)
    
    # Логируем результаты оценки
    writer.add_scalar('Evaluation/ROUGE-1', results['rouge_scores']['rouge1'], 0)
    writer.add_scalar('Evaluation/ROUGE-2', results['rouge_scores']['rouge2'], 0)
    writer.add_scalar('Evaluation/ROUGE-L', results['rouge_scores']['rougeL'], 0)
    
    # Добавляем примеры текстов
    for i, example in enumerate(results['examples'][:3]):  # Первые 3 примера
        writer.add_text(f'Example_{i+1}/Source', example['source'], 0)
        writer.add_text(f'Example_{i+1}/Reference', example['reference'], 0)
        writer.add_text(f'Example_{i+1}/Generated', example['generated'], 0)
    
    writer.close()
    print(f"Evaluation results logged to TensorBoard: {eval_log_dir}")
    
    # Сохранение примеров предсказаний
    with open('predictions_on_test.txt', 'w', encoding='utf-8') as f:
        for i, example in enumerate(results['examples']):
            f.write(f"Example {i+1}:\n")
            f.write(f"Source: {example['source']}\n")
            f.write(f"Reference: {example['reference']}\n")
            f.write(f"Generated: {example['generated']}\n")
            f.write("-" * 80 + "\n")
    
    print("Predictions saved to predictions_on_test.txt")
    
    # Создание примеров визуализации внимания
    print("Creating attention visualization examples...")
    create_attention_examples(model, word_field, test_iter, device, num_examples=3)
    
    # Тестирование на собственных примерах
    print("\nTesting on custom examples:")
    custom_examples = [
        "Президент России Владимир Путин провел встречу с министрами правительства для обсуждения экономической ситуации в стране.",
        "Новый закон о цифровых правах граждан вступит в силу с первого января следующего года.",
        "Ученые обнаружили новый вид бактерий в глубинах океана, который может помочь в борьбе с загрязнением.",
        "Московский метрополитен объявил о запуске новой линии, которая соединит центр города с аэропортом.",
        "Российская сборная по футболу одержала победу в товарищеском матче против команды Бразилии со счетом 2:1."
    ]
    
    # Создаем отдельный writer для пользовательских примеров
    custom_writer = SummaryWriter(f"runs/custom_examples_{timestamp}")
    
    for i, example in enumerate(custom_examples):
        generated, _ = generate_summary(model, word_field, example, device=device)
        print(f"\nCustom Example {i+1}:")
        print(f"Source: {example}")
        print(f"Generated: {generated}")
        
        # Логируем в TensorBoard
        custom_writer.add_text(f'Custom_Example_{i+1}/Source', example, 0)
        custom_writer.add_text(f'Custom_Example_{i+1}/Generated', generated, 0)
    
    custom_writer.close()
    print(f"\nCustom examples logged to TensorBoard: runs/custom_examples_{timestamp}")
    print(f"Run 'tensorboard --logdir=runs' to view all logs")


if __name__ == "__main__":
    main() 