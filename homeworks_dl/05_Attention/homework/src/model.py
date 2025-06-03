# from seminar_code: Transformer architecture with shared embeddings (Task 4)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fasttext
import io
import os
from pathlib import Path
from src import get_logger

# Создаем логгер для этого модуля
logger = get_logger(__name__)

class SharedEmbeddings(nn.Module):
    """
    Общие эмбеддинги для энкодера, декодера и выходного слоя (Задание 4).
    Поддерживает предобученные русские эмбеддинги FastText (Задание 6).
    """
    def __init__(self, vocab_size, d_model, use_pretrained=False, fasttext_path=None, field=None):
        super().__init__()
        self.d_model = d_model
        self.use_pretrained = use_pretrained
        
        if use_pretrained and fasttext_path and field:
            logger.info(f"Loading pretrained embeddings from {fasttext_path}...")
            self.embedding = self._load_pretrained_embeddings(vocab_size, d_model, fasttext_path, field)
            logger.info("✓ Pretrained embeddings loaded successfully!")
        else:
            # Обычные рандомные эмбеддинги
            self.embedding = nn.Embedding(vocab_size, d_model)
            # Инициализация весов как в семинаре
            nn.init.normal_(self.embedding.weight, mean=0, std=d_model**-0.5)
    
    def _load_pretrained_embeddings(self, vocab_size, d_model, fasttext_path, field):
        """Загружает предобученные FastText эмбеддинги."""
        # Проверяем существование файла
        if not os.path.exists(fasttext_path):
            logger.warning(f"Warning: FastText file {fasttext_path} not found. Using random embeddings.")
            embedding = nn.Embedding(vocab_size, d_model)
            nn.init.normal_(embedding.weight, mean=0, std=d_model**-0.5)
            return embedding
        
        try:
            # Загружаем модель FastText
            ft_model = fasttext.load_model(fasttext_path)
            logger.info(f"Loading pretrained embeddings from {fasttext_path}...")
            
            # Проверяем размерность
            ft_dim = ft_model.get_dimension()
            
            # Проверяем совместимость размерностей
            if ft_dim != d_model:
                logger.warning(f"FastText dimension ({ft_dim}) != model dimension ({d_model})")
                logger.warning("Using random embeddings instead.")
                # Возвращаемся к случайным эмбеддингам
                embedding = nn.Embedding(vocab_size, d_model)
                nn.init.normal_(embedding.weight, mean=0, std=d_model**-0.5)
                return embedding
            
            # Создаем матрицу эмбеддингов
            embedding_matrix = torch.zeros(vocab_size, d_model)
            found_words = 0
            
            # Заполняем матрицу для слов из словаря
            for idx, word in enumerate(field.vocab.itos):
                if idx >= vocab_size:
                    break
                try:
                    # Получаем вектор слова
                    vector = ft_model.get_word_vector(word)
                    embedding_matrix[idx] = torch.tensor(vector, dtype=torch.float32)
                    found_words += 1
                except:
                    # Если слово не найдено, используем случайный вектор
                    embedding_matrix[idx] = torch.randn(d_model) * (d_model**-0.5)
            
            logger.info(f"✓ Found embeddings for {found_words}/{vocab_size} words ({100*found_words/vocab_size:.1f}%)")
            
            # Создаем embedding слой с предзагруженными весами
            embedding = nn.Embedding(vocab_size, d_model)
            embedding.weight.data.copy_(embedding_matrix)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error loading FastText model: {e}")
            logger.warning("Using random embeddings instead.")
            # Fallback на случайные эмбеддинги
            embedding = nn.Embedding(vocab_size, d_model)
            nn.init.normal_(embedding.weight, mean=0, std=d_model**-0.5)
            return embedding
    
    def forward(self, x):
        """Применяет эмбеддинги с масштабированием."""
        return self.embedding(x) * math.sqrt(self.d_model)
    
    def get_output_weights(self):
        """Возвращает веса для выходного слоя (transpose эмбеддингов)."""
        return self.embedding.weight


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer."""
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LayerNorm(nn.Module):
    """Layer Normalization."""
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)
        return self.a_2 * (inputs - mean) / (std + self.eps) + self.b_2


class ResidualBlock(nn.Module):
    """Residual connection с layer norm."""
    def __init__(self, size, dropout_rate):
        super().__init__()
        self._norm = LayerNorm(size)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, sublayer):
        return inputs + self._dropout(sublayer(self._norm(inputs)))


class ScaledDotProductAttention(nn.Module):
    """Механизм внимания с масштабированием."""
    def __init__(self, dropout_rate):
        super().__init__()
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self._dropout(attention_weights)
        
        return torch.matmul(attention_weights, value), attention_weights


class MultiHeadedAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, heads_count, d_model, dropout_rate=0.1):
        super().__init__()
        assert d_model % heads_count == 0
        
        self.d_k = d_model // heads_count
        self.heads_count = heads_count
        
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout_rate)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [
            l(x).view(batch_size, -1, self.heads_count, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # Расширяем маску для multi-head attention
        if mask is not None:
            # mask: (batch, 1, seq) или (batch, seq, seq) -> (batch, heads, seq, seq)
            if mask.dim() == 3 and mask.size(1) == 1:
                # Для source mask: (batch, 1, seq) -> (batch, heads, 1, seq)
                mask = mask.unsqueeze(1).repeat(1, self.heads_count, 1, 1)
            elif mask.dim() == 3:
                # Для target mask: (batch, seq, seq) -> (batch, heads, seq, seq)
                mask = mask.unsqueeze(1).repeat(1, self.heads_count, 1, 1)

        x, attn = self.attention(query, key, value, mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads_count * self.d_k)

        return self.output_linear(x), attn


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed forward network."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        return self.w_2(self.dropout(F.relu(self.w_1(inputs))))


class EncoderBlock(nn.Module):
    """Блок энкодера."""
    def __init__(self, size, self_attn, feed_forward, dropout_rate):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([ResidualBlock(size, dropout_rate) for _ in range(2)])
        self.size = size

    def forward(self, inputs, mask):
        x = self.sublayer[0](inputs, lambda x: self.self_attn(x, x, x, mask)[0])
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """Transformer Encoder."""
    def __init__(self, shared_embeddings, d_model, d_ff, blocks_count, heads_count, dropout_rate):
        super().__init__()
        self.embeddings = shared_embeddings
        self.pe = PositionalEncoding(d_model, dropout_rate)
        
        layer = EncoderBlock(
            d_model, 
            MultiHeadedAttention(heads_count, d_model, dropout_rate),
            PositionwiseFeedForward(d_model, d_ff, dropout_rate), 
            dropout_rate
        )
        self.layers = nn.ModuleList([layer for _ in range(blocks_count)])
        self.norm = LayerNorm(d_model)

    def forward(self, inputs, mask):
        x = self.pe(self.embeddings(inputs))
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Слой декодера."""
    def __init__(self, size, self_attn, encoder_attn, feed_forward, dropout_rate):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.encoder_attn = encoder_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([ResidualBlock(size, dropout_rate) for _ in range(3)])

    def forward(self, inputs, encoder_output, source_mask, target_mask):
        m = encoder_output
        x = self.sublayer[0](inputs, lambda x: self.self_attn(x, x, x, target_mask)[0])
        x = self.sublayer[1](x, lambda x: self.encoder_attn(x, m, m, source_mask)[0])
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """Transformer Decoder."""
    def __init__(self, shared_embeddings, d_model, d_ff, blocks_count, heads_count, dropout_rate):
        super().__init__()
        self.embeddings = shared_embeddings
        self.pe = PositionalEncoding(d_model, dropout_rate)
        
        layer = DecoderLayer(
            d_model,
            MultiHeadedAttention(heads_count, d_model, dropout_rate),
            MultiHeadedAttention(heads_count, d_model, dropout_rate),
            PositionwiseFeedForward(d_model, d_ff, dropout_rate),
            dropout_rate
        )
        self.layers = nn.ModuleList([layer for _ in range(blocks_count)])
        self.norm = LayerNorm(d_model)

    def forward(self, inputs, encoder_output, source_mask, target_mask):
        x = self.pe(self.embeddings(inputs))
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)


class Generator(nn.Module):
    """Выходной слой, использующий общие эмбеддинги (Задание 4)."""
    def __init__(self, d_model, shared_embeddings):
        super().__init__()
        self.shared_embeddings = shared_embeddings
        self.proj = nn.Linear(d_model, shared_embeddings.embedding.num_embeddings)
        # Привязываем веса к общим эмбеддингам
        self.proj.weight = shared_embeddings.embedding.weight

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class TransformerSummarizer(nn.Module):
    """Полная модель Transformer для суммаризации."""
    def __init__(self, source_vocab_size, target_vocab_size, d_model=256, d_ff=1024,
                 blocks_count=4, heads_count=8, dropout_rate=0.1, use_pretrained=False, 
                 fasttext_path=None, field=None):
        super().__init__()
        
        # Общие эмбеддинги для всех компонентов (Задание 4)
        assert source_vocab_size == target_vocab_size, "Используем один словарь для source и target"
        self.shared_embeddings = SharedEmbeddings(
            source_vocab_size, d_model, use_pretrained, fasttext_path, field
        )
        
        self.encoder = Encoder(self.shared_embeddings, d_model, d_ff, blocks_count, heads_count, dropout_rate)
        self.decoder = Decoder(self.shared_embeddings, d_model, d_ff, blocks_count, heads_count, dropout_rate)
        self.generator = Generator(d_model, self.shared_embeddings)

    def forward(self, source_inputs, target_inputs, source_mask, target_mask):
        encoder_output = self.encoder(source_inputs, source_mask)
        decoder_output = self.decoder(target_inputs, encoder_output, source_mask, target_mask)
        return self.generator(decoder_output)
    
    def encode(self, source_inputs, source_mask):
        """Энкодинг источника."""
        return self.encoder(source_inputs, source_mask)
    
    def decode(self, target_inputs, encoder_output, source_mask, target_mask):
        """Декодинг с учетом выхода энкодера."""
        return self.decoder(target_inputs, encoder_output, source_mask, target_mask)


def create_model(vocab_size, d_model=256, d_ff=1024, blocks_count=4, heads_count=8, 
                 dropout_rate=0.1, use_pretrained=False, fasttext_path=None, field=None):
    """
    Создает модель Transformer для суммаризации.
    
    Args:
        vocab_size (int): Размер словаря
        d_model (int): Размерность модели (должна быть 300 для FastText)
        d_ff (int): Размерность feed-forward слоя
        blocks_count (int): Количество блоков в энкодере и декодере
        heads_count (int): Количество голов в multi-head attention
        dropout_rate (float): Вероятность dropout
        use_pretrained (bool): Использовать предобученные эмбеддинги (Задание 6)
        fasttext_path (str): Путь к файлу FastText
        field (torchtext.Field): Поле для доступа к словарю
        
    Returns:
        TransformerSummarizer: Модель для суммаризации
    """
    model = TransformerSummarizer(
        source_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        blocks_count=blocks_count,
        heads_count=heads_count,
        dropout_rate=dropout_rate,
        use_pretrained=use_pretrained,
        fasttext_path=fasttext_path,
        field=field
    )
    
    # Инициализация параметров (кроме предзагруженных эмбеддингов)
    for name, p in model.named_parameters():
        if p.dim() > 1 and not (use_pretrained and 'embedding' in name):
            nn.init.xavier_uniform_(p)
    
    return model


def create_model_with_pretrained_embeddings(vocab_size, field, fasttext_path='src/embeddings/cc.ru.300.bin'):
    """
    Удобная функция для создания модели с предобученными русскими эмбеддингами (Задание 6).
    
    Args:
        vocab_size (int): Размер словаря
        field (torchtext.Field): Поле для доступа к словарю
        fasttext_path (str): Путь к файлу FastText
        
    Returns:
        TransformerSummarizer: Модель с предобученными эмбеддингами
    """
    return create_model(
        vocab_size=vocab_size,
        d_model=300,  # FastText размерность
        d_ff=1024,
        blocks_count=4,
        heads_count=6,  # 300 должно делиться на количество голов
        dropout_rate=0.1,
        use_pretrained=True,
        fasttext_path=fasttext_path,
        field=field
    ) 