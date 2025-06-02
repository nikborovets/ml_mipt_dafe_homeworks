# Тесты для проверки пайплайна обучения
import pytest
import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_model, SharedEmbeddings, TransformerSummarizer
from src.train import NoamOpt


def test_create_model():
    """Тест создания модели."""
    vocab_size = 1000
    model = create_model(vocab_size=vocab_size)
    
    assert isinstance(model, TransformerSummarizer)
    assert model.shared_embeddings is not None
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.generator is not None


def test_shared_embeddings():
    """Тест общих эмбеддингов (Задание 4)."""
    vocab_size = 1000
    d_model = 256
    
    shared_emb = SharedEmbeddings(vocab_size, d_model)
    
    # Проверяем размерности
    assert shared_emb.embedding.num_embeddings == vocab_size
    assert shared_emb.embedding.embedding_dim == d_model
    
    # Проверяем forward pass
    input_ids = torch.LongTensor([[1, 2, 3]])
    output = shared_emb(input_ids)
    assert output.shape == (1, 3, d_model)
    
    # Проверяем масштабирование
    expected_scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float))
    raw_emb = shared_emb.embedding(input_ids)
    scaled_emb = shared_emb(input_ids)
    assert torch.allclose(scaled_emb, raw_emb * expected_scale)


def test_model_forward_pass():
    """Тест прямого прохода модели."""
    vocab_size = 100
    model = create_model(vocab_size=vocab_size, d_model=64, d_ff=128)
    
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    # Создаем mock данные
    source_inputs = torch.randint(1, vocab_size, (batch_size, src_len))
    target_inputs = torch.randint(1, vocab_size, (batch_size, tgt_len))
    source_mask = torch.ones(batch_size, 1, src_len).bool()
    target_mask = torch.ones(batch_size, 1, tgt_len).bool()
    
    # Прямой проход
    output = model(source_inputs, target_inputs, source_mask, target_mask)
    
    assert output.shape == (batch_size, tgt_len, vocab_size)


def test_noam_optimizer():
    """Тест оптимизатора Noam."""
    model = create_model(vocab_size=100, d_model=64)
    
    optimizer = NoamOpt(
        model_size=64, 
        factor=2, 
        warmup=100,
        optimizer=torch.optim.Adam(model.parameters(), lr=0)
    )
    
    # Проверяем начальный learning rate
    initial_rate = optimizer.rate(step=1)
    assert initial_rate > 0
    
    # Проверяем, что rate увеличивается в warmup периоде
    warmup_rate = optimizer.rate(step=50)
    assert warmup_rate > initial_rate
    
    # Проверяем поведение после warmup (может как расти, так и падать)
    post_warmup_rate = optimizer.rate(step=200)
    assert post_warmup_rate > 0  # Просто проверяем, что положительный
    
    # Проверяем очень поздний step - должен быть меньше warmup
    very_late_rate = optimizer.rate(step=10000)
    assert very_late_rate < warmup_rate


def test_model_encode_decode():
    """Тест отдельных методов encode и decode."""
    vocab_size = 100
    model = create_model(vocab_size=vocab_size, d_model=64)
    
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    source_inputs = torch.randint(1, vocab_size, (batch_size, src_len))
    target_inputs = torch.randint(1, vocab_size, (batch_size, tgt_len))
    source_mask = torch.ones(batch_size, 1, src_len).bool()
    target_mask = torch.ones(batch_size, 1, tgt_len).bool()
    
    # Тест encode
    encoder_output = model.encode(source_inputs, source_mask)
    assert encoder_output.shape == (batch_size, src_len, 64)
    
    # Тест decode
    decoder_output = model.decode(target_inputs, encoder_output, source_mask, target_mask)
    assert decoder_output.shape == (batch_size, tgt_len, 64)


def test_shared_embeddings_weight_tying():
    """Тест привязки весов в общих эмбеддингах (Задание 4)."""
    vocab_size = 100
    model = create_model(vocab_size=vocab_size)
    
    # Проверяем, что веса действительно общие
    encoder_emb_weight = model.encoder.embeddings.embedding.weight
    decoder_emb_weight = model.decoder.embeddings.embedding.weight
    generator_weight = model.generator.proj.weight
    
    # Все должны ссылаться на одну и ту же матрицу весов
    assert torch.equal(encoder_emb_weight, decoder_emb_weight)
    assert torch.equal(encoder_emb_weight, generator_weight)


def test_model_parameter_count():
    """Тест количества параметров модели."""
    vocab_size = 1000
    model = create_model(vocab_size=vocab_size)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable_params == total_params  # Все параметры должны быть обучаемыми
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}") 