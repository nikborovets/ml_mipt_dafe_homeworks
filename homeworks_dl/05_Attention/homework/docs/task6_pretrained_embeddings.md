# Задание 6: Предобученные русские эмбеддинги

## Описание

Задание 6 предполагает замену случайных эмбеддингов на предобученные русские эмбеддинги FastText и сравнение качества модели до и после замены.

## Реализация

### 1. Загрузка русских эмбеддингов FastText

Создан скрипт `src/embeddings/download_embeddings.py` для автоматической загрузки:

```bash
python -m src.embeddings.download_embeddings
```

Скачивает официальные русские эмбеддинги:
- **Источник**: https://fasttext.cc/docs/en/crawl-vectors.html
- **URL**: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz
- **Размерность**: 300
- **Обучено на**: Common Crawl + Wikipedia

### 2. Модификация класса SharedEmbeddings

Класс `SharedEmbeddings` в `src/model.py` расширен для поддержки предобученных эмбеддингов:

```python
class SharedEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, use_pretrained=False, 
                 fasttext_path=None, field=None):
        # ... инициализация предобученных весов
```

**Особенности реализации**:
- Автоматическая проверка существования файла FastText
- Сопоставление слов из словаря с эмбеддингами FastText
- Fallback на случайные эмбеддинги при ошибках
- Статистика покрытия словаря предобученными векторами

### 3. Автоматическое сравнение

Обучение автоматически определяет наличие предобученных эмбеддингов:

**С предобученными эмбеддингами:**
```bash
python -m src.train  # если cc.ru.300.bin существует
```
- Использует d_model=300 (размерность FastText)
- Heads=6 (300 должно делиться на количество голов)
- Сохраняет как `best_model_pretrained.pt`

**Без предобученных эмбеддингов:**
```bash
python -m src.train  # если cc.ru.300.bin отсутствует
```
- Использует d_model=256 (случайные эмбеддинги)  
- Heads=8
- Сохраняет как `best_model_random.pt`

### 4. Структура выходных файлов

```
.
├── best_model_pretrained.pt      # Модель с предобученными эмбеддингами
├── best_model_random.pt          # Модель со случайными эмбеддингами
├── training_plot_pretrained.png  # Графики обучения (предобученные)
├── training_plot_random.png      # Графики обучения (случайные)
├── model_info_pretrained.json    # Метаданные модели (предобученные)
├── model_info_random.json        # Метаданные модели (случайные)
└── src/embeddings/
    └── cc.ru.300.bin             # Файл русских эмбеддингов FastText
```

## Использование

### Шаг 1: Загрузка эмбеддингов
```bash
# Активируем виртуальное окружение
source .venv/bin/activate

# Устанавливаем зависимости
pip install -r requirements.txt

# Загружаем русские эмбеддинги (~1.5 GB)
python -m src.embeddings.download_embeddings
```

### Шаг 2: Обучение с предобученными эмбеддингами
```bash
# Запускаем обучение (автоматически использует предобученные эмбеддинги)
python -m src.train
```

### Шаг 3: Сравнение с базовой моделью
```bash
# Переименовываем или удаляем файл эмбеддингов
mv src/embeddings/cc.ru.300.bin src/embeddings/cc.ru.300.bin.backup

# Обучаем модель со случайными эмбеддингами
python -m src.train

# Возвращаем эмбеддинги обратно
mv src/embeddings/cc.ru.300.bin.backup src/embeddings/cc.ru.300.bin
```

## Ожидаемые результаты

**Метрики для сравнения:**
- ROUGE-1, ROUGE-2, ROUGE-L scores
- Validation loss
- Скорость сходимости
- Качество генерируемых суммаризаций

**Гипотеза:** Предобученные русские эмбеддинги должны показать:
- Лучшие ROUGE метрики
- Более быструю сходимость
- Более низкий validation loss
- Более качественные суммаризации

## Технические детали

### Совместимость размерностей
- **FastText размерность**: 300
- **Количество голов**: 6 (300 % 6 == 0)
- **Feed-forward размерность**: 1024

### Покрытие словаря
Скрипт выводит статистику покрытия:
```
✓ Found embeddings for 45123/55785 words (80.9%)
```

### Обработка отсутствующих слов
Для слов, не найденных в FastText:
- Инициализация случайными векторами N(0, d_model^(-0.5))
- Продолжение обучения всех эмбеддингов (freeze=False)

## Команды для воспроизведения

```bash
# Полный цикл Task 6
source .venv/bin/activate
python -m src.embeddings.download_embeddings
python -m src.train  # с предобученными эмбеддингами

# Сравнение с базовой моделью
mv src/embeddings/cc.ru.300.bin src/embeddings/cc.ru.300.bin.backup
python -m src.train  # со случайными эмбеддингами

# Анализ результатов
cat model_info_pretrained.json
cat model_info_random.json
``` 