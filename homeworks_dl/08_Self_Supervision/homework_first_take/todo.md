# TODO - Prototypical Networks для Omniglot

**Дедлайн:** 31.05.2025

## Основные задачи

### 1. Подготовка проекта
- [x] Создать структуру директорий (src/, tests/, models/, metrics/, runs/)
- [x] Создать requirements.txt с зависимостями
- [x] Создать Makefile с командами
- [x] Создать src/__init__.py

### 2. Реализация модулей

#### 2.1 src/data.py
- [x] Перенести функции read_alphabets и read_images из omniglot_hw_from_ipynb.py
- [x] Добавить функции extract_sample и display_sample
- [x] Реализовать аугментации через Albumentations (поворот 90°, 180°, 270°)
- [x] Добавить нормализацию изображений

#### 2.2 src/model.py  
- [x] Создать CNN энкодер (4 блока Conv+BN+ReLU+MaxPool)
- [x] Реализовать класс ProtoNet
- [x] Добавить метод set_forward_loss в ProtoNet

#### 2.3 src/train.py
- [x] Перенести функцию train из omniglot_hw_from_ipynb.py
- [x] Добавить TensorBoard логирование (SummaryWriter)
- [x] Настроить Adam оптимизатор и LR scheduler
- [x] Сохранение лучших весов в models/protonet.pt

#### 2.4 src/evaluate.py
- [x] Перенести функцию test из omniglot_hw_from_ipynb.py
- [x] Добавить функцию visualize_predictions
- [x] Сохранение результатов в metrics/test_accuracy.txt
- [x] Сохранение визуализаций в metrics/

### 3. Тесты (tests/)
- [x] tests/test_dataset.py - тесты датасета
- [x] tests/test_pipeline.py - end-to-end тесты
- [x] tests/test_metrics.py - тесты метрик
- [x] tests/test_loss.py - тесты функции потерь

### 4. DVC Pipeline
- [ ] Создать dvc.yaml с этапами prepare, train, evaluate
- [ ] Настроить отслеживание данных и артефактов

### 5. Финализация
- [ ] Создать usage.md с инструкциями
- [ ] Создать config.yaml для параметров
- [ ] Генерация predictions_on_test.txt (10 примеров)
- [ ] Создать package.sh для архивирования
- [x] Проверить работу `make test`

## Новые задачи
- [x] Создать config.yaml с параметрами
- [ ] Загрузить датасет Omniglot
- [x] Проверить совместимость с устройством mps (M1 Mac)
- [x] Создать базовые тесты для проверки работоспособности

## Приоритет
1. Высокий: src/data.py, src/model.py - основа функциональности
2. Средний: src/train.py, src/evaluate.py - тренировка и оценка
3. Низкий: тесты, DVC, документация 