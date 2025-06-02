.PHONY: install test train evaluate clean package tensorboard train-pretrained train-random evaluate-pretrained evaluate-random pipeline-pretrained pipeline-random

# Установка зависимостей
install:
	source .venv/bin/activate && pip install -r requirements.txt

# Запуск тестов
test:
	source .venv/bin/activate && python -m pytest tests/ -v

# Обработка данных
preprocess:
	source .venv/bin/activate && python src/data.py

# Обучение модели (обычное)
train:
	source .venv/bin/activate && python -m src.train

# Обучение с предобученными эмбеддингами (DVC)
train-pretrained:
	source .venv/bin/activate && dvc repro -s train_pretrained

# Обучение со случайными эмбеддингами (DVC)
train-random:
	source .venv/bin/activate && dvc repro -s train_random

# Оценка модели (обычная)
evaluate:
	source .venv/bin/activate && python -m src.evaluate

# Оценка модели с предобученными эмбеддингами (DVC)
evaluate-pretrained:
	source .venv/bin/activate && dvc repro -s evaluate_pretrained

# Оценка модели со случайными эмбеддингами (DVC)
evaluate-random:
	source .venv/bin/activate && dvc repro -s evaluate_random

# Запуск TensorBoard
tensorboard:
	@echo "Starting TensorBoard server..."
	@echo "Open http://localhost:6006 in your browser"
	source .venv/bin/activate && tensorboard --logdir=runs --port=6006

# Полный пайплайн (оба типа обучения)
pipeline: preprocess train-pretrained train-random evaluate-pretrained evaluate-random

# Быстрый пайплайн (только предобученные эмбеддинги)
pipeline-fast: preprocess train-pretrained evaluate-pretrained

# Пайплайн только для предобученных эмбеддингов (через DVC)
pipeline-pretrained:
	source .venv/bin/activate && dvc repro -s evaluate_pretrained

# Пайплайн только для случайных эмбеддингов (через DVC)
pipeline-random:
	source .venv/bin/activate && dvc repro -s evaluate_random

# Очистка временных файлов
clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -f *.pyc src/*.pyc tests/*.pyc
	rm -rf runs  # Очистка TensorBoard логов

# Создание архива для сдачи
package:
	bash package.sh

# Запуск DVC пайплайна (все стадии)
dvc-repro:
	source .venv/bin/activate && dvc repro

# Инициализация DVC
dvc-init:
	source .venv/bin/activate && dvc init 