# 📊 Результаты обучения модели Transformer для суммаризации

**Автор:** Borovets N.V.  
**Задание:** HW03 - Transformer Summarization (русские новости)

## 🎯 Основные результаты

### ✅ Выполненные задания:
1. **Генерация суммаризации** - реализовано
2. **ROUGE метрики** - реализовано  
3. **Визуализация attention** - реализовано
4. **Общие эмбеддинги** - реализовано
5. **Label Smoothing** - реализовано
6. **Предобученные русские эмбеддинги** - реализовано

### 📈 Финальные метрики модели:
Все метрики сохранены в соответствующих JSON файлах в директориях результатов.

## 📂 Где найти результаты

### 🗂️ Структура результатов:
```
# Модели
best_model_pretrained.pt             # Лучшая модель с предобученными эмбеддингами
best_model_random.pt                 # Лучшая модель со случайными эмбеддингами

# Результаты оценки
evaluation_results/
├── 100epoch_train_pretrained_20250602_220847/  # Результаты модели с предобученными эмбеддингами
│   ├── rouge_scores_pretrained.json
│   ├── predictions_on_test_pretrained.txt
│   ├── model_info_pretrained.json
│   ├── training_plot_pretrained.png
│   └── attention_examples_pretrained/          # Матрицы внимания
└── 100epoch_train_random_20250603_075020/      # Результаты модели со случайными эмбеддингами
    ├── rouge_scores_random.json
    ├── predictions_on_test_random.txt
    ├── model_info_random.json
    ├── training_plot_random.png
    └── attention_examples_random/              # Матрицы внимания
```

### 📊 Просмотр всех метрик через TensorBoard:

```bash
# Запуск TensorBoard для просмотра всех метрик
tensorboard --logdir=runs
или
make tensorboard

# Откроется в браузере по адресу: http://localhost:6006
```

**В TensorBoard доступно:**
- 📈 **Графики loss** (train/validation по эпохам и батчам)
- 📊 **ROUGE метрики** (ROUGE-1, ROUGE-2, ROUGE-L)
- 🧠 **Learning rate** и градиенты
- 📝 **Кастомные примеры** суммаризации на русском языке
- 🎯 **Архитектура модели** (граф вычислений)

## 📋 Технические детали

- **Архитектура**: Transformer (4 слоя, 8 голов внимания)
- **Данные**: русские новости из news.csv
- **Loss**: Label Smoothing KL-Divergence (0.1 smoothing)
- **Оптимизатор**: Noam scheduler с warmup
- **Предобученные эмбеддинги**: FastText Russian (300d)

## 💡 Интерпретация результатов

- **ROUGE метрики**: доступны в JSON файлах результатов
- **Loss значения**: отображены в графиках training_plot_*.png  
- **Модель обучается**: видно снижение loss и рост ROUGE по эпохам
- **Attention работает**: визуализации показывают осмысленные паттерны внимания

---
*Для полной документации см. `docs/usage.md`* 