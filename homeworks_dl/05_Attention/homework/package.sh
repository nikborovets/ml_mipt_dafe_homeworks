#!/bin/bash

# Скрипт для создания архива домашнего задания
echo "Creating submission archive..."

# Проверяем наличие необходимых файлов
if [ ! -f "best_model.pt" ]; then
    echo "Warning: best_model.pt not found. Run training first."
fi

if [ ! -f "training_plot.png" ]; then
    echo "Warning: training_plot.png not found. Run training first."
fi

if [ ! -f "predictions_on_test.txt" ]; then
    echo "Warning: predictions_on_test.txt not found. Run evaluation first."
fi

# Создаем архив
zip -r BorovetsNV-hw03.zip \
    src/ \
    tests/ \
    docs/ \
    best_model.pt \
    training_plot.png \
    predictions_on_test.txt \
    requirements.txt \
    todo.md \
    Makefile \
    dvc.yaml \
    -x "*.pyc" "*__pycache__*" "*.DS_Store"

echo "Archive BorovetsNV-hw03.zip created successfully!"
echo "Contents:"
unzip -l BorovetsNV-hw03.zip 