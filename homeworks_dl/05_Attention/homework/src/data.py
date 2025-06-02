# from seminar_code: data processing and tokenization
import pandas as pd
import torch
import pickle
import os
from torchtext.data import Field, Example, Dataset, BucketIterator
from tqdm.auto import tqdm
from . import get_device

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'


def load_and_process_data(csv_path='news.csv', train_ratio=0.85, min_freq=7, device=None):
    """
    Загружает и обрабатывает данные для обучения модели суммаризации.
    
    Args:
        csv_path (str): Путь к CSV файлу с данными
        train_ratio (float): Доля данных для обучения
        min_freq (int): Минимальная частота слова для включения в словарь
        device (str/torch.device): Устройство для PyTorch. Если None, выбирается автоматически
    
    Returns:
        tuple: (train_iter, test_iter, word_field)
    """
    if device is None:
        device = get_device()
    
    word_field = Field(tokenize='moses', init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
    fields = [('source', word_field), ('target', word_field)]
    
    data = pd.read_csv(csv_path, delimiter=',')
    
    examples = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing data"):
        source_text = word_field.preprocess(row.text)
        target_text = word_field.preprocess(row.title)
        examples.append(Example.fromlist([source_text, target_text], fields))
    
    dataset = Dataset(examples, fields)
    train_dataset, test_dataset = dataset.split(split_ratio=train_ratio)
    
    print(f'Train size = {len(train_dataset)}')
    print(f'Test size = {len(test_dataset)}')
    
    word_field.build_vocab(train_dataset, min_freq=min_freq)
    print(f'Vocab size = {len(word_field.vocab)}')
    
    train_iter, test_iter = BucketIterator.splits(
        datasets=(train_dataset, test_dataset), 
        batch_sizes=(16, 32), 
        shuffle=True, 
        device=device, 
        sort=False
    )
    
    return train_iter, test_iter, word_field


def make_mask(source_inputs, target_inputs, pad_idx):
    """
    Создает маски для источника и цели.
    
    Args:
        source_inputs: Входные данные источника
        target_inputs: Входные данные цели
        pad_idx: Индекс токена паддинга
        
    Returns:
        tuple: (source_mask, target_mask)
    """
    # from seminar_code: mask creation
    source_mask = (source_inputs != pad_idx).unsqueeze(-2)
    
    target_mask = (target_inputs != pad_idx).unsqueeze(-2)
    target_mask = target_mask & subsequent_mask(target_inputs.size(-1)).type_as(target_mask.data)
    
    return source_mask, target_mask


def subsequent_mask(size):
    """
    Создает маску для предотвращения просмотра будущих токенов.
    
    Args:
        size (int): Размер последовательности
        
    Returns:
        torch.Tensor: Маска размера (1, size, size)
    """
    # from seminar_code: subsequent mask for decoder
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def convert_batch(batch, pad_idx=1):
    """
    Конвертирует батч данных для обучения.
    
    Args:
        batch: Батч данных от DataLoader
        pad_idx: Индекс токена паддинга
        
    Returns:
        tuple: (source, target, source_mask, target_mask)
    """
    # from seminar_code: batch conversion
    source, target = batch.source.transpose(0, 1), batch.target.transpose(0, 1)
    
    source_inputs = source[:, :-1] 
    target_inputs = target[:, :-1]
    target_outputs = target[:, 1:]
    
    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx)
    
    return source_inputs, target_inputs, target_outputs, source_mask, target_mask


def load_processed_data(data_dir='data_processed'):
    """
    Загружает обработанные данные из директории.
    
    Args:
        data_dir (str): Путь к директории с обработанными данными
        
    Returns:
        tuple: (vocab, stats) где vocab - словарь, stats - статистика
    """
    vocab_path = os.path.join(data_dir, 'vocab.pkl')
    stats_path = os.path.join(data_dir, 'stats.pkl')
    
    if not os.path.exists(vocab_path) or not os.path.exists(stats_path):
        raise FileNotFoundError(f"Processed data not found in {data_dir}. Run preprocess stage first.")
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    return vocab, stats


def load_processed_data_iterators(data_dir='data_processed', csv_path='news.csv', train_ratio=0.85, device=None):
    """
    Загружает обработанные данные и создает итераторы без повторной обработки.
    
    Args:
        data_dir (str): Путь к директории с обработанными данными
        csv_path (str): Путь к исходному CSV файлу
        train_ratio (float): Доля данных для обучения
        device (str/torch.device): Устройство для PyTorch
        
    Returns:
        tuple: (train_iter, test_iter, word_field)
    """
    if device is None:
        device = get_device()
    
    # Загружаем словарь и статистики
    vocab, stats = load_processed_data(data_dir)
    
    # Создаем field с загруженным словарем
    word_field = Field(tokenize='moses', init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
    word_field.vocab = vocab
    
    # Создаем поля для данных
    fields = [('source', word_field), ('target', word_field)]
    
    # Загружаем исходные данные и создаем examples
    data = pd.read_csv(csv_path, delimiter=',')
    
    examples = []
    for _, row in data.iterrows():
        source_text = word_field.preprocess(row.text)
        target_text = word_field.preprocess(row.title)
        examples.append(Example.fromlist([source_text, target_text], fields))
    
    dataset = Dataset(examples, fields)
    train_dataset, test_dataset = dataset.split(split_ratio=train_ratio)
    
    # Создаем итераторы
    train_iter, test_iter = BucketIterator.splits(
        datasets=(train_dataset, test_dataset), 
        batch_sizes=(16, 32), 
        shuffle=True, 
        device=device, 
        sort=False
    )
    
    return train_iter, test_iter, word_field


if __name__ == "__main__":
    # Тестовый запуск обработки данных
    device = get_device()
    train_iter, test_iter, word_field = load_and_process_data(device=device)
    
    print("Data processing completed successfully!")
    print(f"Vocabulary size: {len(word_field.vocab)}")
    print(f"PAD token index: {word_field.vocab.stoi['<pad>']}")
    
    # Сохраняем обработанные данные для DVC
    os.makedirs('data_processed', exist_ok=True)
    
    # Сохраняем словарь
    with open('data_processed/vocab.pkl', 'wb') as f:
        pickle.dump(word_field.vocab, f)
    
    # Сохраняем статистику
    stats = {
        'vocab_size': len(word_field.vocab),
        'train_size': len(train_iter.dataset),
        'test_size': len(test_iter.dataset),
        'pad_token_idx': word_field.vocab.stoi['<pad>'],
        'bos_token_idx': word_field.vocab.stoi[BOS_TOKEN],
        'eos_token_idx': word_field.vocab.stoi[EOS_TOKEN],
    }
    
    with open('data_processed/stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"Processed data saved to data_processed/") 