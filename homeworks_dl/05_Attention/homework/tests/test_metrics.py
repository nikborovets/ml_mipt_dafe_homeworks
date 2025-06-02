# Тесты для проверки ROUGE метрик
import pytest

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import compute_rouge_scores


def test_compute_rouge_scores_identical():
    """Тест ROUGE метрик для идентичных текстов."""
    generated_texts = ["hello world", "this is a test"]
    reference_texts = ["hello world", "this is a test"]
    
    scores = compute_rouge_scores(generated_texts, reference_texts)
    
    assert 'rouge1' in scores
    assert 'rouge2' in scores
    assert 'rougeL' in scores
    
    # Для идентичных текстов все метрики должны быть близки к 1.0
    assert scores['rouge1'] > 0.9
    assert scores['rouge2'] > 0.9
    assert scores['rougeL'] > 0.9


def test_compute_rouge_scores_different():
    """Тест ROUGE метрик для разных текстов."""
    generated_texts = ["hello world", "completely different text"]
    reference_texts = ["goodbye world", "this is a test"]
    
    scores = compute_rouge_scores(generated_texts, reference_texts)
    
    assert 'rouge1' in scores
    assert 'rouge2' in scores
    assert 'rougeL' in scores
    
    # Для разных текстов метрики должны быть низкими
    assert 0 <= scores['rouge1'] <= 1
    assert 0 <= scores['rouge2'] <= 1
    assert 0 <= scores['rougeL'] <= 1


def test_compute_rouge_scores_partial_overlap():
    """Тест ROUGE метрик для частично пересекающихся текстов."""
    generated_texts = ["hello world test"]
    reference_texts = ["hello world example"]
    
    scores = compute_rouge_scores(generated_texts, reference_texts)
    
    # Должно быть частичное совпадение
    assert scores['rouge1'] > 0.5  # 2 из 3 слов совпадают
    assert scores['rouge2'] > 0  # Есть биграмма "hello world"
    assert scores['rougeL'] > 0.5


def test_compute_rouge_scores_empty():
    """Тест ROUGE метрик для пустых списков."""
    generated_texts = []
    reference_texts = []
    
    scores = compute_rouge_scores(generated_texts, reference_texts)
    
    # Для пустых списков должны быть NaN или 0
    assert 'rouge1' in scores
    assert 'rouge2' in scores
    assert 'rougeL' in scores


def test_compute_rouge_scores_single_word():
    """Тест ROUGE метрик для одного слова."""
    generated_texts = ["hello"]
    reference_texts = ["hello"]
    
    scores = compute_rouge_scores(generated_texts, reference_texts)
    
    assert scores['rouge1'] == 1.0  # Полное совпадение
    # ROUGE-2 может быть 0 для одного слова (нет биграмм)
    assert scores['rougeL'] == 1.0  # Полное совпадение


def test_compute_rouge_scores_case_sensitivity():
    """Тест чувствительности к регистру."""
    generated_texts = ["Hello World"]
    reference_texts = ["hello world"]
    
    scores = compute_rouge_scores(generated_texts, reference_texts)
    
    # ROUGE scorer обычно нечувствителен к регистру
    assert scores['rouge1'] > 0.9
    assert scores['rougeL'] > 0.9 