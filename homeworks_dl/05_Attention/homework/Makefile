.PHONY: install test train evaluate clean package tensorboard

# Установка зависимостей
install:
	source .venv/bin/activate && pip install -r requirements.txt

# Запуск тестов
test:
	source .venv/bin/activate && python -m pytest tests/ -v

# Обработка данных
preprocess:
	source .venv/bin/activate && python src/data.py

# Обучение модели
train:
	source .venv/bin/activate && python -m src.train

# Оценка модели
evaluate:
	source .venv/bin/activate && python -m src.evaluate

# Запуск TensorBoard
tensorboard:
	@echo "Starting TensorBoard server..."
	@echo "Open http://localhost:6006 in your browser"
	source .venv/bin/activate && tensorboard --logdir=runs --port=6006

# Полный пайплайн
pipeline: preprocess train evaluate

# Очистка временных файлов
clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -f *.pyc src/*.pyc tests/*.pyc
	rm -rf runs  # Очистка TensorBoard логов

# Создание архива для сдачи
package:
	bash package.sh

# Запуск DVC пайплайна
dvc-repro:
	source .venv/bin/activate && dvc repro

# Инициализация DVC
dvc-init:
	source .venv/bin/activate && dvc init 