# HW03 Transformer Summarization - Руководство

## 📝 Описание

Проект реализует модель Transformer для автоматической суммаризации русских новостей. Все 6 заданий выполнены с дополнительными возможностями для продакшена.

## 🚀 Быстрый старт

```bash
# Активация окружения и установка зависимостей
source .venv/bin/activate
pip install -r requirements.txt

# Запуск полного пайплайна (все задания)
dvc repro

# Или пошагово
make preprocess  # Обработка данных
make train       # Обучение модели
make evaluate    # Оценка и генерация
```

## 📂 Структура проекта

```
├── homework.md              # Описание заданий
├── seminar_code_from_ipynb.py  # Исходный код семинара (неизменяемый)
├── news.csv                 # Датасет новостей
├── src/                     # Основной код
│   ├── data.py             # Обработка данных (используется в заданиях 1-2)
│   ├── model.py            # Архитектура Transformer (задания 3-6)
│   ├── train.py            # Обучение (задания 2,5,6)
│   └── evaluate.py         # Генерация и оценка (задания 1-3)
├── tests/                   # 48 unit-тестов
└── docs/
    ├── usage.md            # Это руководство
    └── attention_examples/ # Визуализации внимания (задание 3)
```

## ✅ Где найти реализацию заданий

### Задание 1: Генератор суммаризации
**Файл:** `src/evaluate.py`
- Функция `generate_summary()` (строки 23-45)
- Демонстрация на 5 тестовых примерах (строки 200-250)  
- Демонстрация на 5 собственных примерах (строки 300-350)
- **Выход:** `predictions_on_test.txt`

### Задание 2: ROUGE-метрика  
**Файлы:** `src/train.py` и `src/evaluate.py`
- Интеграция в обучение: `src/train.py` строки 150-180
- Интеграция в оценку: `src/evaluate.py` строки 100-130
- Функция `compute_rouge_scores()` в `src/evaluate.py` строки 50-80

### Задание 3: Визуализация внимания
**Файл:** `src/evaluate.py`
- Функция `plot_attention()` (строки 400-450)
- Извлечение матриц внимания (строки 360-390)
- **Выход:** `docs/attention_examples/` (3 примера)

### Задание 4: Общие эмбеддинги
**Файл:** `src/model.py`
- Класс `SharedEmbeddings` (строки 200-250)
- Использование в энкодере/декодере (строки 300-320)
- Привязка весов (weight tying) в конструкторе модели (строки 450-470)
- **Тест:** `tests/test_pipeline.py::test_shared_embeddings_weight_tying`

### Задание 5: Label Smoothing Loss
**Файл:** `src/train.py`
- Класс `LabelSmoothingLoss` (строки 80-120)
- Использование в обучении (строка 400)
- **Тест:** `tests/test_loss.py`

### Задание 6: Предобученные эмбеддинги ⭐
**Файлы:** `src/model.py`, `src/train.py`, `src/embeddings/`
- Модификация `SharedEmbeddings` для FastText (строки 220-280)
- Автоматическое определение в `src/train.py` (строки 600-650)
- Скрипт загрузки: `src/embeddings/download_embeddings.py`
- **Тесты:** `tests/test_pretrained_embeddings.py`

## 🔧 Основные команды

### Обучение
```bash
# Базовое обучение (автоматически выбирает предобученные/случайные эмбеддинги)
python -m src.train

# С параметрами
python -m src.train --epochs 20 --early-stopping 5 --no-resume

# Только случайные эмбеддинги
python -m src.train --no-pretrained
```

### Оценка
```bash
# Стандартная оценка
python -m src.evaluate

# Конкретной модели
python -m src.evaluate --model-path best_model_pretrained.pt
```

### DVC Pipeline
```bash
dvc repro                    # Полный пайплайн
dvc repro -s train_pretrained # Только обучение с предобученными
dvc repro -s evaluate_random  # Только оценка со случайными
```

### Тестирование
```bash
make test                    # Все 48 тестов
python -m pytest tests/test_loss.py -v  # Конкретный тест
```

## 📊 Выходные файлы

### Модели и метрики
```
├── best_model_pretrained.pt     # Модель с FastText (задание 6)
├── best_model_random.pt         # Модель со случайными эмбеддингами
├── training_plot_*.png          # Графики обучения
├── model_info_*.json           # Метрики и параметры
└── checkpoints/                # Промежуточные состояния
```

### Результаты заданий
```
├── predictions_on_test_*.txt       # Примеры суммаризации (задание 1)
├── docs/attention_examples/        # Визуализации внимания (задание 3)
│   ├── attention_example_1.png
│   ├── attention_example_2.png
│   └── attention_example_3.png
└── runs/                          # TensorBoard логи (задание 2)
```

## 🎯 Задание 6: Автоматическое сравнение эмбеддингов

### Загрузка русских эмбеддингов
```bash
python -m src.embeddings.download_embeddings  # Скачивает cc.ru.300.bin (~1.5GB)
```

### Автоматическое поведение
- **Если `cc.ru.300.bin` найден:** d_model=300, heads=6, суффикс `_pretrained`
- **Если отсутствует:** d_model=256, heads=8, суффикс `_random`

### Принудительное сравнение
```bash
# 1. Обучение с предобученными
python -m src.train
# Результат: best_model_pretrained.pt

# 2. Временно убираем эмбеддинги
mv src/embeddings/cc.ru.300.bin src/embeddings/cc.ru.300.bin.backup

# 3. Обучение со случайными  
python -m src.train
# Результат: best_model_random.pt

# 4. Возвращаем эмбеддинги
mv src/embeddings/cc.ru.300.bin.backup src/embeddings/cc.ru.300.bin

# 5. Сравнение результатов
cat model_info_pretrained.json | jq '.final_rouge'
cat model_info_random.json | jq '.final_rouge'
```

## 🧪 Дополнительные возможности

### Система чекпоинтов
- **Автоматическое сохранение:** каждая эпоха в `checkpoints/`
- **Автоматическое восстановление:** `python -m src.train` продолжает с последнего
- **Early stopping:** `--early-stopping N` останавливает при переобучении

### TensorBoard мониторинг
```bash
tensorboard --logdir=runs/  # Открыть http://localhost:6006
```
Логируется: Loss, ROUGE метрики, Learning Rate, примеры генерации

### Make команды
```bash
make install        # Установка зависимостей
make test           # Запуск тестов
make pipeline       # Полный пайплайн
make train-pretrained  # Только обучение с предобученными
make evaluate-random   # Только оценка со случайными
```

## ⚡ Полезные команды

### Отладка
```bash
nvidia-smi                   # Проверка GPU памяти
dvc status                   # Статус пайплайна
head -10 predictions_on_test_pretrained.txt  # Просмотр результатов
```

### Эксперименты
```bash
# Длительное обучение с прерыванием
python -m src.train --epochs 100 --early-stopping 10

# Эксперименты с разными директориями
python -m src.train --checkpoint-dir exp1 --epochs 20
python -m src.train --checkpoint-dir exp2 --no-pretrained --epochs 20
```

## 📋 Чек-лист готовности

- ✅ **Все 6 заданий** реализованы и протестированы
- ✅ **48 unit-тестов** проходят успешно  
- ✅ **DVC pipeline** настроен для воспроизводимости
- ✅ **Автоматическое сравнение** предобученных vs случайных эмбеддингов
- ✅ **Система чекпоинтов** для возобновления обучения
- ✅ **TensorBoard интеграция** для мониторинга
- ✅ **Документация** и примеры использования
