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

## Запуск полного pipeline через DVC

```bash
dvc repro
```

Это выполнит все этапы:
1. Обработка данных (`data.py`)
2. Обучение модели (`train.py`)
3. Оценка модели (`evaluate.py`)

## Ручной запуск отдельных этапов

### 1. Обработка данных
```bash
python -m src.data
```

### 2. Обучение модели с чекпоинтами
```bash
# Базовое обучение (10 эпох)
python -m src.train

# Обучение с параметрами
python -m src.train --epochs 20 --early-stopping 5

# Начать обучение заново (игнорировать чекпоинты)
python -m src.train --no-resume

# Использовать только случайные эмбеддинги
python -m src.train --no-pretrained

# Сохранять чекпоинт каждые 2 эпохи
python -m src.train --save-every 2

# Указать свою директорию для чекпоинтов
python -m src.train --checkpoint-dir my_checkpoints
```

### Параметры обучения

- `--epochs N`: Количество эпох (по умолчанию 10)
- `--no-resume`: Начать обучение заново, игнорируя существующие чекпоинты
- `--checkpoint-dir DIR`: Директория для сохранения чекпоинтов (по умолчанию `checkpoints`)
- `--save-every N`: Сохранять чекпоинт каждые N эпох (по умолчанию 1)
- `--early-stopping N`: Остановка при отсутствии улучшений N эпох
- `--no-pretrained`: Принудительно использовать случайные эмбеддинги

### 3. Оценка модели
```bash
python -m src.evaluate
```

## Механизм чекпоинтов

Проект поддерживает полноценную систему чекпоинтов для возобновления обучения:

### Автоматическое сохранение
- Чекпоинты сохраняются после каждой эпохи (настраивается `--save-every`)
- Автоматически сохраняется `latest_checkpoint.pth` - последний чекпоинт
- Лучшая модель по validation loss сохраняется в `best_model.pth`

### Автоматическое возобновление
- При запуске `python -m src.train` автоматически ищется и загружается последний чекпоинт
- Обучение продолжается с того места, где остановилось
- Сохраняется состояние оптимизатора, learning rate scheduler, история обучения

### Структура чекпоинта
```python
{
    'epoch': 5,                           # Номер эпохи
    'model_state_dict': model.state_dict(), # Веса модели
    'optimizer_state_dict': optimizer.state_dict(), # Состояние оптимизатора  
    'optimizer_step': 1500,               # Шаг оптимизатора NoamOpt
    'optimizer_rate': 0.0001,             # Текущий learning rate
    'history': {...},                     # История обучения (losses, ROUGE)
    'timestamp': '2024-01-15T10:30:00'    # Время создания
}
```

### Управление чекпоинтами
- Автоматически удаляются старые чекпоинты (сохраняются последние 3)
- `latest_checkpoint.pth` всегда содержит самый свежий чекпоинт
- `best_model.pth` содержит модель с лучшим validation loss

### Примеры использования

**Длительное обучение с возможностью прерывания:**
```bash
# Запуск обучения на 50 эпох
python -m src.train --epochs 50

# Если обучение прервалось, просто запустите снова:
python -m src.train --epochs 50
# Обучение продолжится с того места, где остановилось
```

**Экспериментирование с параметрами:**
```bash
# Первый эксперимент
python -m src.train --epochs 10 --checkpoint-dir exp1

# Второй эксперимент  
python -m src.train --epochs 10 --checkpoint-dir exp2 --no-pretrained

# Возврат к первому эксперименту
python -m src.train --epochs 20 --checkpoint-dir exp1
```

**Early stopping:**
```bash
# Остановка при отсутствии улучшений 5 эпох подряд
python -m src.train --epochs 100 --early-stopping 5
```

## Предобученные эмбеддинги (Задание 6)

Проект поддерживает использование предобученных русских эмбеддингов FastText:

### Загрузка эмбеддингов
```bash
python -m src.embeddings.download_embeddings
```

### Автоматическое определение
- Если найден файл `src/embeddings/cc.ru.300.bin`, используются предобученные эмбеддинги
- Иначе используются случайные эмбеддинги

### Принудительное использование случайных эмбеддингов
```bash
python -m src.train --no-pretrained
```

## Мониторинг обучения

### TensorBoard
```bash
# Запуск TensorBoard (логи сохраняются в runs/)
tensorboard --logdir runs

# Открыть http://localhost:6006 в браузере
```

В TensorBoard доступны:
- Графики loss по эпохам и батчам
- ROUGE метрики
- Learning rate schedule
- Гистограммы весов и градиентов модели
- Граф модели

### Сохраняемые файлы

После обучения создаются файлы:
- `best_model_*.pt` - веса лучшей модели
- `training_plot_*.png` - графики обучения  
- `model_info_*.json` - метаданные и результаты
- `checkpoints/` - директория с чекпоинтами

## Тестирование

```bash
# Запуск всех тестов
make test

# Запуск конкретного теста
python -m pytest tests/test_pipeline.py::test_checkpoint_functionality -v
```

## Задания

Проект реализует все 6 заданий:

1. ✅ **Генератор суммаризации** - функция `generate_summary()` в `evaluate.py`
2. ✅ **ROUGE-метрика** - интегрирована в обучение и оценку
3. ✅ **Визуализация внимания** - функция `plot_attention()` в `evaluate.py`  
4. ✅ **Общие эмбеддинги** - класс `SharedEmbeddings` в `model.py`
5. ✅ **Label Smoothing Loss** - класс `LabelSmoothingLoss` в `train.py`
6. ✅ **Предобученные эмбеддинги** - автоматическое определение и использование FastText

### Дополнительно реализовано:
- ✅ **Система чекпоинтов** - полное сохранение/восстановление состояния обучения
- ✅ **Early stopping** - автоматическая остановка при переобучении  
- ✅ **TensorBoard интеграция** - мониторинг метрик в реальном времени
- ✅ **Автоматический выбор устройства** - mps → cuda → cpu
- ✅ **Полное тестирование** - 48 unit-тестов с покрытием всего функционала 