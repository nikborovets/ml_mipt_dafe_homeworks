# Руководство по использованию HW03 Transformer Summarization

## Описание проекта

Этот проект реализует модель Transformer для автоматической суммаризации русских новостных текстов. Модель обучается генерировать заголовки по текстам новостей.

## Структура проекта

```
.
├── homework.md              # Описание домашнего задания
├── todo.md                 # Чек-лист выполнения задач
├── seminar_code_from_ipynb.py  # Исходный код семинара (не изменяется)
├── requirements.txt        # Зависимости Python
├── Makefile               # Автоматизация команд
├── dvc.yaml              # DVC пайплайн
├── news.csv              # Датасет новостей
├── src/                  # Исходный код проекта
│   ├── __init__.py
│   ├── data.py          # Обработка данных
│   ├── model.py         # Архитектура Transformer
│   ├── train.py         # Обучение с Label Smoothing
│   └── evaluate.py      # Генерация и оценка
├── tests/               # Тесты
│   ├── test_dataset.py
│   ├── test_pipeline.py
│   ├── test_metrics.py
│   └── test_loss.py
└── docs/               # Документация
    ├── usage.md
    └── attention_examples/  # Примеры визуализации внимания
```

## Установка и настройка

### 1. Активация виртуального окружения

```bash
source .venv/bin/activate
```

### 2. Установка зависимостей

```bash
make install
# или
pip install -r requirements.txt
```

## Использование

### Полный пайплайн

Запуск всего пайплайна (обработка данных → обучение → оценка):

```bash
make pipeline
```

### Пошаговое выполнение

#### 1. Обработка данных

```bash
make preprocess
# или
python src/data.py
```

#### 2. Обучение модели

```bash
make train
# или
python src/train.py
```

Результаты:
- `best_model.pt` - веса обученной модели
- `training_plot.png` - графики обучения

#### 3. Оценка модели

```bash
make evaluate
# или
python src/evaluate.py
```

Результаты:
- `predictions_on_test.txt` - примеры предсказаний
- `docs/attention_examples/` - визуализации внимания
- ROUGE метрики в консоли

### Тестирование

Запуск всех тестов:

```bash
make test
# или
python -m pytest tests/ -v
```

### DVC пайплайн

Инициализация DVC:

```bash
make dvc-init
```

Запуск DVC пайплайна:

```bash
make dvc-repro
```

## Мониторинг с TensorBoard

### Запуск TensorBoard сервера

```bash
make tensorboard
```

Затем откройте браузер и перейдите по адресу: http://localhost:6006

### Что логируется в TensorBoard

**Во время обучения (`src/train.py`):**
- `Loss/train_epoch` - потери на обучающей выборке по эпохам
- `Loss/val_epoch` - потери на валидационной выборке по эпохам  
- `Loss/train_batch` - потери по батчам (каждые 100 батчей)
- `ROUGE/train_rouge1`, `ROUGE/train_rouge2`, `ROUGE/train_rougeL` - ROUGE метрики на обучении
- `ROUGE/val_rouge1`, `ROUGE/val_rouge2`, `ROUGE/val_rougeL` - ROUGE метрики на валидации
- `Learning_Rate` - изменение learning rate по шагам
- `Parameters/*` - гистограммы весов модели (каждые 5 эпох)
- `Gradients/*` - гистограммы градиентов (каждые 5 эпох)
- Граф модели для визуализации архитектуры

**Во время оценки (`src/evaluate.py`):**
- `Evaluation/ROUGE-1`, `Evaluation/ROUGE-2`, `Evaluation/ROUGE-L` - финальные ROUGE метрики
- `Example_*/Source`, `Example_*/Reference`, `Example_*/Generated` - примеры текстов
- `Custom_Example_*/Source`, `Custom_Example_*/Generated` - результаты на пользовательских примерах

### Структура логов

```
runs/
├── transformer_summarization_YYYYMMDD_HHMMSS/  # Логи обучения
├── evaluation_YYYYMMDD_HHMMSS/                 # Логи оценки
└── custom_examples_YYYYMMDD_HHMMSS/             # Логи пользовательских примеров
```

## Реализованные задания

### Задание 1: Генератор суммаризации
- ✅ Функция `generate_summary()` в `src/evaluate.py`
- ✅ Демонстрация на 5 примерах из теста
- ✅ Демонстрация на 5 собственных примерах

### Задание 2: ROUGE метрики
- ✅ Интеграция `rouge_scorer.RougeScorer`
- ✅ Расчет ROUGE во время обучения
- ✅ Расчет ROUGE при оценке

### Задание 3: Визуализация внимания
- ✅ Извлечение матриц attention
- ✅ Функция `plot_attention()` для визуализации
- ✅ Примеры для 3 случаев в `docs/attention_examples/`

### Задание 4: Общие эмбеддинги
- ✅ Класс `SharedEmbeddings`
- ✅ Использование в энкодере, декодере и выходном слое
- ✅ Привязка весов (weight tying)

### Задание 5: Label Smoothing
- ✅ Класс `LabelSmoothingLoss`
- ✅ Замена стандартного NLL Loss
- ✅ Тесты для проверки корректности

### Задание 6: Предобученные эмбеддинги
- 🔄 В процессе реализации (требует MCP context7)

## Архитектура модели

Модель основана на архитектуре Transformer:

- **Энкодер**: 4 слоя, 8 голов внимания
- **Декодер**: 4 слоя, 8 голов внимания  
- **Размерность модели**: 256
- **Размерность FFN**: 1024
- **Общие эмбеддинги**: для энкодера, декодера и выходного слоя

## Особенности реализации

1. **Label Smoothing**: Сглаживание меток с коэффициентом 0.1
2. **Noam Optimizer**: Адаптивное расписание learning rate
3. **ROUGE метрики**: Автоматический расчет во время обучения
4. **Визуализация внимания**: Heatmap для анализа механизма внимания
5. **Общие эмбеддинги**: Экономия параметров и улучшение обобщения

## Результаты

После обучения модель генерирует файлы:

- `best_model.pt` - веса модели
- `training_plot.png` - графики loss и ROUGE метрик
- `predictions_on_test.txt` - примеры суммаризации
- `docs/attention_examples/` - визуализации внимания

## Создание архива для сдачи

```bash
make package
```

Создает архив `BorovetsNV-hw03.zip` со всеми необходимыми файлами.

## Требования к системе

- Python 3.11+
- PyTorch 1.9+
- 8GB+ RAM для обучения
- GPU/MPS рекомендуется для ускорения 