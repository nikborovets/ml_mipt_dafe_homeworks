# Тесты для проверки датасета
import pytest
import torch
import tempfile
import os
import shutil
import pickle
import pandas as pd
from torchtext.data import Field

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import get_device
from src.data import load_and_process_data, convert_batch, make_mask, subsequent_mask, load_processed_data, load_processed_data_iterators


def test_load_and_process_data():
    """Тест загрузки и обработки данных."""
    device = get_device()
    
    # Создаем временный CSV файл для тестирования с правильной структурой как в news.csv
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(",text,title\n")
        f.write('1,"Президент России Владимир Путин провел встречу с министрами правительства для обсуждения экономической ситуации в стране. На совещании были рассмотрены вопросы развития промышленности и поддержки малого бизнеса в условиях санкций.",Путин обсудил экономику с министрами\n')
        f.write('2,"Новый закон о цифровых правах граждан вступит в силу с первого января следующего года. Документ регулирует использование персональных данных в интернете и защищает права пользователей социальных сетей.",Закон о цифровых правах вступит в силу\n')
        temp_csv = f.name
    
    try:
        train_iter, test_iter, word_field = load_and_process_data(
            csv_path=temp_csv, train_ratio=0.5, min_freq=1, device=device
        )
        
        # Проверяем, что итераторы созданы
        assert train_iter is not None
        assert test_iter is not None
        assert word_field is not None
        
        # Проверяем словарь
        assert len(word_field.vocab) > 0
        assert '<pad>' in word_field.vocab.stoi
        assert '<s>' in word_field.vocab.stoi
        assert '</s>' in word_field.vocab.stoi
        
    finally:
        os.unlink(temp_csv)


def test_convert_batch():
    """Тест конвертации батча."""
    # Создаем mock batch
    class MockBatch:
        def __init__(self):
            self.source = torch.LongTensor([[1, 2, 3, 0], [4, 5, 0, 0]]).transpose(0, 1)
            self.target = torch.LongTensor([[1, 2, 3, 0], [4, 5, 6, 0]]).transpose(0, 1)
    
    batch = MockBatch()
    source_inputs, target_inputs, target_outputs, source_mask, target_mask = convert_batch(batch, pad_idx=0)
    
    assert source_inputs.shape[1] == 3  # Убираем последний токен
    assert target_inputs.shape[1] == 3  # Убираем последний токен
    assert target_outputs.shape[1] == 3  # Сдвигаем на 1
    assert source_mask is not None
    assert target_mask is not None


def test_make_mask():
    """Тест создания масок."""
    source_inputs = torch.LongTensor([[1, 2, 3], [4, 5, 0]])
    target_inputs = torch.LongTensor([[1, 2, 3], [4, 0, 0]])
    
    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx=0)
    
    assert source_mask.shape == (2, 1, 3)
    assert target_mask.shape == (2, 3, 3)  # Треугольная маска из-за subsequent_mask


def test_subsequent_mask():
    """Тест маски для предотвращения просмотра будущих токенов."""
    mask = subsequent_mask(4)
    
    assert mask.shape == (1, 4, 4)
    # Проверяем, что маска треугольная
    assert mask[0, 0, 1] == False  # Не можем смотреть в будущее
    assert mask[0, 1, 0] == True   # Можем смотреть в прошлое


def test_vocabulary_tokens():
    """Тест наличия специальных токенов в словаре."""
    _, _, word_field = load_and_process_data(csv_path='news.csv', device=get_device())
    
    vocab = word_field.vocab
    assert '<s>' in vocab.stoi
    assert '</s>' in vocab.stoi
    assert '<pad>' in vocab.stoi
    assert '<unk>' in vocab.stoi


def test_save_and_load_processed_data():
    """Тест сохранения и загрузки обработанных данных."""
    device = get_device()
    
    # Создаем временную директорию
    with tempfile.TemporaryDirectory() as temp_dir:
        # Создаем временный CSV файл с правильной структурой
        csv_path = os.path.join(temp_dir, 'test_data.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(",text,title\n")
            f.write('1,"Московский метрополитен объявил о запуске новой линии, которая соединит центр города с аэропортом. Строительство планируется завершить к концу следующего года.",Метро запустит новую линию к аэропорту\n')
            f.write('2,"Российская сборная по футболу одержала победу в товарищеском матче против команды Бразилии со счетом 2:1. Матч прошел на стадионе в Москве при полных трибунах.",Россия обыграла Бразилию 2:1\n')
            f.write('3,"Ученые обнаружили новый вид бактерий в глубинах океана, который может помочь в борьбе с загрязнением пластиком.",Найден новый вид бактерий против пластика\n')
        
        # Обрабатываем данные
        train_iter, test_iter, word_field = load_and_process_data(
            csv_path=csv_path, train_ratio=0.6, min_freq=1, device=device
        )
        
        # Сохраняем обработанные данные
        data_processed_dir = os.path.join(temp_dir, 'data_processed')
        os.makedirs(data_processed_dir, exist_ok=True)
        
        # Сохраняем словарь
        vocab_path = os.path.join(data_processed_dir, 'vocab.pkl')
        with open(vocab_path, 'wb') as f:
            pickle.dump(word_field.vocab, f)
        
        # Сохраняем статистику
        stats = {
            'vocab_size': len(word_field.vocab),
            'train_size': len(train_iter.dataset),
            'test_size': len(test_iter.dataset),
            'pad_token_idx': word_field.vocab.stoi['<pad>'],
            'bos_token_idx': word_field.vocab.stoi['<s>'],
            'eos_token_idx': word_field.vocab.stoi['</s>'],
        }
        stats_path = os.path.join(data_processed_dir, 'stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        
        # Загружаем обработанные данные
        loaded_vocab, loaded_stats = load_processed_data(data_processed_dir)
        
        # Проверяем, что данные загрузились корректно
        assert len(loaded_vocab) == len(word_field.vocab)
        assert loaded_stats['vocab_size'] == len(word_field.vocab)
        assert loaded_stats['pad_token_idx'] == word_field.vocab.stoi['<pad>']
        assert loaded_stats['bos_token_idx'] == word_field.vocab.stoi['<s>']
        assert loaded_stats['eos_token_idx'] == word_field.vocab.stoi['</s>']


def test_load_processed_data_iterators():
    """Тест загрузки итераторов из обработанных данных."""
    device = get_device()
    
    # Создаем временную директорию
    with tempfile.TemporaryDirectory() as temp_dir:
        # Создаем временный CSV файл с правильной структурой
        csv_path = os.path.join(temp_dir, 'test_data.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(",text,title\n")
            f.write('1,"Президент Украины выступил с заявлением о ситуации в экономике страны и планах развития на следующий год.",Президент выступил с заявлением\n')
            f.write('2,"Новое исследование показало эффективность вакцины против гриппа в условиях пандемии коронавируса.",Исследование показало эффективность вакцины\n')
            f.write('3,"Центральный банк принял решение о снижении ключевой ставки на 0,25 процентных пункта.",ЦБ снизил ключевую ставку\n')
            f.write('4,"Министерство образования объявило о реформе системы высшего образования в стране.",Минобразования объявило о реформе\n')
        
        # Сначала обрабатываем и сохраняем данные
        train_iter_orig, test_iter_orig, word_field_orig = load_and_process_data(
            csv_path=csv_path, train_ratio=0.7, min_freq=1, device=device
        )
        
        # Сохраняем обработанные данные
        data_processed_dir = os.path.join(temp_dir, 'data_processed')
        os.makedirs(data_processed_dir, exist_ok=True)
        
        # Сохраняем словарь
        vocab_path = os.path.join(data_processed_dir, 'vocab.pkl')
        with open(vocab_path, 'wb') as f:
            pickle.dump(word_field_orig.vocab, f)
        
        # Сохраняем статистику
        stats = {
            'vocab_size': len(word_field_orig.vocab),
            'train_size': len(train_iter_orig.dataset),
            'test_size': len(test_iter_orig.dataset),
            'pad_token_idx': word_field_orig.vocab.stoi['<pad>'],
            'bos_token_idx': word_field_orig.vocab.stoi['<s>'],
            'eos_token_idx': word_field_orig.vocab.stoi['</s>'],
        }
        stats_path = os.path.join(data_processed_dir, 'stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        
        # Теперь загружаем итераторы из обработанных данных
        train_iter_loaded, test_iter_loaded, word_field_loaded = load_processed_data_iterators(
            data_dir=data_processed_dir, csv_path=csv_path, train_ratio=0.7, device=device
        )
        
        # Проверяем, что итераторы созданы
        assert train_iter_loaded is not None
        assert test_iter_loaded is not None
        assert word_field_loaded is not None
        
        # Проверяем, что словари идентичны
        assert len(word_field_loaded.vocab) == len(word_field_orig.vocab)
        assert word_field_loaded.vocab.stoi['<pad>'] == word_field_orig.vocab.stoi['<pad>']
        assert word_field_loaded.vocab.stoi['<s>'] == word_field_orig.vocab.stoi['<s>']
        assert word_field_loaded.vocab.stoi['</s>'] == word_field_orig.vocab.stoi['</s>']
        
        # Проверяем, что размеры датасетов совпадают
        assert len(train_iter_loaded.dataset) == len(train_iter_orig.dataset)
        assert len(test_iter_loaded.dataset) == len(test_iter_orig.dataset)


def test_load_processed_data_file_not_found():
    """Тест обработки ошибки при отсутствии файлов обработанных данных."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Пытаемся загрузить данные из пустой директории
        with pytest.raises(FileNotFoundError):
            load_processed_data(temp_dir) 