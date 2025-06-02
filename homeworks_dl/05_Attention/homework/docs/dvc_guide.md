# Руководство по DVC и запуску проекта

## Что такое DVC?

**DVC (Data Version Control)** - это система контроля версий для данных и ML пайплайнов. Она позволяет:

- 📊 Версионировать большие файлы данных и модели
- 🔄 Создавать воспроизводимые ML пайплайны  
- 📈 Отслеживать эксперименты и метрики
- 🚀 Автоматически запускать только изменившиеся этапы

## Структура DVC пайплайна в нашем проекте

```
news.csv → preprocess → train → evaluate
   ↓           ↓         ↓        ↓
data_processed/ → best_model.pt → predictions_on_test.txt
                     ↓              ↓
              training_plot.png  attention_examples/
                     ↓
                  runs/ (TensorBoard)
```

## 🚀 Быстрый старт

### 1. Установка и активация окружения

```bash
# Активация виртуального окружения
source .venv/bin/activate

# Установка всех зависимостей (включая DVC)
make install
```

### 2. Инициализация DVC

```bash
# Инициализация DVC в проекте
make dvc-init

# Или напрямую:
dvc init
```

### 3. Запуск полного пайплайна

```bash
# Вариант 1: Через DVC (рекомендуется)
make dvc-repro

# Вариант 2: Через Make
make pipeline

# Вариант 3: Пошагово
make preprocess  # Обработка данных
make train       # Обучение модели  
make evaluate    # Оценка и генерация
```

## 📋 Детальное описание этапов

### Этап 1: `preprocess` - Обработка данных

**Команда:** `python -m src.data`

**Входы:**
- `news.csv` - исходный датасет новостей
- `src/data.py` - код обработки данных

**Выходы:**
- `data_processed/` - обработанные данные (если создаются)

**Что происходит:**
- Загрузка и токенизация данных
- Создание словаря
- Разделение на train/test
- Создание итераторов данных

### Этап 2: `train` - Обучение модели

**Команда:** `python -m src.train`

**Входы:**
- Все файлы из `src/`
- `news.csv` - данные для обучения

**Выходы:**
- `best_model.pt` - веса обученной модели
- `training_plot.png` - графики обучения
- `runs/` - TensorBoard логи

**Что происходит:**
- Создание Transformer модели
- Обучение с Label Smoothing Loss
- Расчет ROUGE метрик
- Сохранение весов и графиков
- Логирование в TensorBoard

### Этап 3: `evaluate` - Оценка модели

**Команда:** `python -m src.evaluate`

**Входы:**
- Исходный код
- `best_model.pt` - обученная модель
- `news.csv` - данные для тестирования

**Выходы:**
- `predictions_on_test.txt` - примеры предсказаний
- `docs/attention_examples/` - визуализации внимания
- `evaluation_results/rouge_scores.json` - метрики

**Что происходит:**
- Загрузка обученной модели
- Генерация суммаризаций на тестовых данных
- Расчет финальных ROUGE метрик
- Создание визуализаций внимания
- Тестирование на пользовательских примерах

## 🔧 DVC команды

### Основные команды

```bash
# Инициализация DVC
dvc init

# Запуск пайплайна (только измененные этапы)
dvc repro

# Принудительный запуск всех этапов
dvc repro --force

# Просмотр статуса пайплайна
dvc status

# Просмотр DAG (направленного графа)
dvc dag

# Показать метрики
dvc metrics show

# Показать различия в метриках
dvc metrics diff
```

### Управление выходными файлами

```bash
# Добавить файл под контроль DVC
dvc add large_file.csv

# Отправить файлы в remote storage
dvc push

# Получить файлы из remote storage  
dvc pull

# Проверить статус файлов
dvc status
```

## 🎯 Различные способы запуска

### 1. Полный автоматический пайплайн (DVC)

```bash
make dvc-repro
```

**Преимущества:**
- ✅ Запускает только измененные этапы
- ✅ Автоматическое отслеживание зависимостей
- ✅ Кэширование результатов
- ✅ Воспроизводимость

### 2. Полный пайплайн (Make)

```bash
make pipeline
```

**Преимущества:**
- ✅ Простота
- ✅ Последовательное выполнение
- ✅ Контроль через Makefile

### 3. Пошаговое выполнение

```bash
# Шаг 1: Обработка данных
make preprocess

# Шаг 2: Обучение (с TensorBoard)
make train

# Шаг 3: Оценка модели
make evaluate

# Опционально: Запуск TensorBoard
make tensorboard
```

### 4. Прямое выполнение модулей

```bash
# Активируем окружение
source .venv/bin/activate

# Запускаем напрямую
python -m src.train
python -m src.evaluate
```

## 📊 Мониторинг и результаты

### TensorBoard

```bash
# Запуск TensorBoard сервера
make tensorboard

# Открыть в браузере
open http://localhost:6006
```

**Доступные метрики:**
- Loss (train/val)
- ROUGE-1/2/L scores
- Learning Rate
- Model parameters histograms
- Model graph visualization

### Результирующие файлы

После завершения пайплайна получите:

```
├── best_model.pt              # Веса модели
├── training_plot.png          # Графики обучения
├── predictions_on_test.txt    # Примеры предсказаний
├── docs/attention_examples/   # Визуализации внимания
├── runs/                      # TensorBoard логи
└── evaluation_results/        # Метрики в JSON
```

## 🔄 Типичные рабочие процессы

### Первый запуск проекта

```bash
# 1. Установка
source .venv/bin/activate
make install

# 2. Инициализация DVC
make dvc-init

# 3. Полный запуск
make dvc-repro

# 4. Мониторинг
make tensorboard
```

### Изменение кода и повторный запуск

```bash
# Если изменили код обучения
# DVC автоматически определит, что нужно переобучить
make dvc-repro

# Если нужно принудительно переобучить
dvc repro --force train
```

### Эксперименты с гиперпараметрами

```bash
# 1. Изменить параметры в src/train.py
# 2. Запустить только обучение
dvc repro train evaluate

# 3. Сравнить метрики
dvc metrics diff
```

## 🐛 Устранение проблем

### Проблема с импортами

```bash
# Убедитесь, что запускаете модули правильно
python -m src.train  # ✅ Правильно
python src/train.py  # ❌ Ошибка импорта
```

### Проблема с зависимостями

```bash
# Переустановка зависимостей
make install

# Проверка установки DVC
dvc version
```

### Проблема с TensorBoard

```bash
# Проверка запущенных процессов
ps aux | grep tensorboard

# Перезапуск TensorBoard
make tensorboard
```

## 🎯 Рекомендации

1. **Используйте DVC** для воспроизводимости экспериментов
2. **Мониторьте через TensorBoard** процесс обучения
3. **Версионируйте изменения** в git + DVC
4. **Документируйте эксперименты** в commit messages
5. **Проверяйте результаты** через метрики и визуализации 