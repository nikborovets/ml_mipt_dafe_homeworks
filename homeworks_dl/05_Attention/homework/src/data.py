# from seminar_code: data processing and tokenization
import pandas as pd
import torch
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


if __name__ == "__main__":
    # Тестовый запуск обработки данных
    device = get_device()
    train_iter, test_iter, word_field = load_and_process_data(device=device)
    
    print("Data processing completed successfully!")
    print(f"Vocabulary size: {len(word_field.vocab)}")
    print(f"PAD token index: {word_field.vocab.stoi['<pad>']}") 