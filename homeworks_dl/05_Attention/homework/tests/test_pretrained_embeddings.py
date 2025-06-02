# Tests for Task 6: Pretrained embeddings functionality
import pytest
import torch
import tempfile
import os
import shutil
from unittest.mock import Mock, patch
from src.model import SharedEmbeddings, create_model_with_pretrained_embeddings, create_model
from src import get_device


def test_shared_embeddings_random():
    """Test SharedEmbeddings with random initialization."""
    vocab_size = 1000
    d_model = 256
    
    embeddings = SharedEmbeddings(vocab_size, d_model, use_pretrained=False)
    
    assert embeddings.embedding.num_embeddings == vocab_size
    assert embeddings.embedding.embedding_dim == d_model
    assert not embeddings.use_pretrained
    
    # Test forward pass
    x = torch.randint(0, vocab_size, (2, 10))
    output = embeddings(x)
    assert output.shape == (2, 10, d_model)


def test_shared_embeddings_pretrained_file_not_found():
    """Test SharedEmbeddings when pretrained file doesn't exist."""
    vocab_size = 1000
    d_model = 300
    
    # Mock field
    mock_field = Mock()
    mock_field.vocab.itos = [f'word_{i}' for i in range(vocab_size)]
    
    embeddings = SharedEmbeddings(
        vocab_size, d_model, 
        use_pretrained=True, 
        fasttext_path='non_existent_file.bin',
        field=mock_field
    )
    
    # Should fallback to random embeddings
    assert embeddings.embedding.num_embeddings == vocab_size
    assert embeddings.embedding.embedding_dim == d_model


@patch('fasttext.load_model')
def test_shared_embeddings_pretrained_success(mock_load_model):
    """Test SharedEmbeddings with successful pretrained loading."""
    vocab_size = 100
    d_model = 300
    
    # Mock FastText model
    mock_ft_model = Mock()
    mock_ft_model.get_dimension.return_value = 300
    mock_ft_model.get_word_vector.return_value = torch.randn(300).numpy()
    mock_load_model.return_value = mock_ft_model
    
    # Mock field
    mock_field = Mock()
    mock_field.vocab.itos = [f'word_{i}' for i in range(vocab_size)]
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.bin') as temp_file:
        embeddings = SharedEmbeddings(
            vocab_size, d_model,
            use_pretrained=True,
            fasttext_path=temp_file.name,
            field=mock_field
        )
        
        assert embeddings.embedding.num_embeddings == vocab_size
        assert embeddings.embedding.embedding_dim == d_model
        assert embeddings.use_pretrained


@patch('fasttext.load_model')
def test_shared_embeddings_dimension_mismatch(mock_load_model):
    """Test SharedEmbeddings with dimension mismatch."""
    vocab_size = 100
    d_model = 256  # Different from FastText dimension
    
    # Mock FastText model with different dimension
    mock_ft_model = Mock()
    mock_ft_model.get_dimension.return_value = 300  # FastText returns 300, but we want 256
    mock_load_model.return_value = mock_ft_model
    
    # Mock field
    mock_field = Mock()
    mock_field.vocab.itos = [f'word_{i}' for i in range(vocab_size)]
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.bin') as temp_file:
        embeddings = SharedEmbeddings(
            vocab_size, d_model,
            use_pretrained=True,
            fasttext_path=temp_file.name,
            field=mock_field
        )
        
        # Should fallback to random embeddings
        assert embeddings.embedding.num_embeddings == vocab_size
        assert embeddings.embedding.embedding_dim == d_model


def test_create_model_with_pretrained_embeddings():
    """Test create_model_with_pretrained_embeddings function."""
    vocab_size = 1000
    
    # Mock field
    mock_field = Mock()
    mock_field.vocab.itos = [f'word_{i}' for i in range(vocab_size)]
    
    # This should use the non-existent path and fallback to random
    model = create_model_with_pretrained_embeddings(
        vocab_size, mock_field, 'non_existent_path.bin'
    )
    
    # Check model structure
    assert hasattr(model, 'shared_embeddings')
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'decoder') 
    assert hasattr(model, 'generator')
    
    # Check dimensions for FastText setup (300d, 6 heads)
    assert model.shared_embeddings.d_model == 300
    assert model.encoder.layers[0].self_attn.heads_count == 6


def test_create_model_regular():
    """Test regular create_model function."""
    vocab_size = 1000
    
    model = create_model(vocab_size=vocab_size, d_model=256)
    
    # Check model structure
    assert hasattr(model, 'shared_embeddings')
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'decoder')
    assert hasattr(model, 'generator')
    
    # Check dimensions
    assert model.shared_embeddings.d_model == 256


def test_create_model_with_pretrained_parameters():
    """Test create_model with pretrained parameters."""
    vocab_size = 1000
    
    # Mock field
    mock_field = Mock()
    mock_field.vocab.itos = [f'word_{i}' for i in range(vocab_size)]
    
    model = create_model(
        vocab_size=vocab_size,
        d_model=300,
        heads_count=6,  # 300 % 6 == 0
        use_pretrained=True,
        fasttext_path='non_existent_path.bin',
        field=mock_field
    )
    
    assert model.shared_embeddings.d_model == 300
    assert model.shared_embeddings.use_pretrained == True
    assert model.encoder.layers[0].self_attn.heads_count == 6


def test_embedding_forward_pass():
    """Test forward pass through embeddings."""
    vocab_size = 1000
    d_model = 256
    batch_size = 4
    seq_len = 20
    
    embeddings = SharedEmbeddings(vocab_size, d_model, use_pretrained=False)
    
    # Create input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = embeddings(x)
    
    # Check output shape and scaling
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Check that it's scaled (should be larger than basic embedding)
    basic_output = embeddings.embedding(x)
    assert torch.allclose(output, basic_output * (d_model ** 0.5))


def test_get_output_weights():
    """Test get_output_weights method."""
    vocab_size = 1000
    d_model = 256
    
    embeddings = SharedEmbeddings(vocab_size, d_model, use_pretrained=False)
    
    output_weights = embeddings.get_output_weights()
    
    # Should return the same weights as embedding layer
    assert torch.equal(output_weights, embeddings.embedding.weight)
    assert output_weights.shape == (vocab_size, d_model) 